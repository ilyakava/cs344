//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.

      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly -
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

// set this to an even number
#define JACOBI_ITR 800
#include "utils.h"
#include <thrust/host_vector.h>

__global__ void
poisson_equation_jacobi_iteration(float* const ImageGuess_next, const float* const ImageGuess_prev,
                                  const unsigned char* const source, const unsigned char* const target,
                                  const unsigned char* const d_sourceMaskInteriorMap, const unsigned char* const d_sourceMask,
                                  const size_t numRowsSource, const size_t numColsSource)
{
  const int2 thread_2D_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_id.x >= numColsSource || thread_2D_id.y >= numRowsSource)
    return;
  const int thread_1D_id = thread_2D_id.y * numColsSource + thread_2D_id.x;
  if (d_sourceMask[thread_1D_id] == 0)
    return;

  const unsigned char spx = source[thread_1D_id];
  char neighbors[4][2] = {
    {0,1},
    {1,0},
    {0,-1},
    {-1,0}
  };
  int Sum1 = 0;
  int Sum2 = 0;

  // Note: no neighbor bounds checking since the mask is assumed to not flow
  // off the edge of the image
  for (char i = 0; i < 4; i++) {
    int neighbor_1D_id = (thread_2D_id.x + neighbors[i][0]) + (thread_2D_id.y + neighbors[i][1]) * numColsSource;
    if (d_sourceMaskInteriorMap[neighbor_1D_id] == 4)
      Sum1 += ImageGuess_prev[neighbor_1D_id];
    else
      Sum1 += target[neighbor_1D_id];
    Sum2 += spx - source[neighbor_1D_id];
  }

  float newVal = (Sum1 + Sum2) / 4.0f;
  ImageGuess_next[thread_1D_id] = fmin(255.0f, fmax(0.0f, newVal));
}

__global__ void
map_source_to_mask(const uchar4* const d_sourceImg, unsigned char* const d_sourceMask,
                   const size_t numRowsSource, const size_t numColsSource)
{
  const int2 thread_2D_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_id.x >= numColsSource || thread_2D_id.y >= numRowsSource)
    return;
  const int thread_1D_id = thread_2D_id.y * numColsSource + thread_2D_id.x;

  const uchar4 px = d_sourceImg[thread_1D_id];
  int brightness = px.x + px.y + px.z;
  if (brightness == 765)
    d_sourceMask[thread_1D_id] = 1;
}

__global__ void
stencil_2d_von_neumann(const unsigned char* const d_in, unsigned char* const d_out,
                       const size_t numRows, const size_t numCols)
{
  const int2 thread_2D_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_id.x >= numCols || thread_2D_id.y >= numRows)
    return;
  const int thread_1D_id = thread_2D_id.y * numCols + thread_2D_id.x;

  // 0,1
  if (thread_2D_id.y)
    d_out[thread_1D_id] += d_in[(thread_2D_id.y-1) * numCols + thread_2D_id.x];
  // 1,0
  if (thread_2D_id.x < (numCols-1))
    d_out[thread_1D_id] += d_in[thread_2D_id.y * numCols + (thread_2D_id.x+1)];
  // 0,-1
  if (thread_2D_id.y < (numRows-1))
    d_out[thread_1D_id] += d_in[(thread_2D_id.y+1) * numCols + thread_2D_id.x];
  // -1,0
  if (thread_2D_id.x)
    d_out[thread_1D_id] += d_in[thread_2D_id.y * numCols + (thread_2D_id.x-1)];
}

__global__ void
separateChannels(const uchar4* const inputImageRGBA,
                 const size_t numRows,
                 const size_t numCols,
                 unsigned char* const redChannel,
                 unsigned char* const greenChannel,
                 unsigned char* const blueChannel)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if ( thread_2D_pos.y >= numRows || thread_2D_pos.x >= numCols )
    return;
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  const uchar4 px = inputImageRGBA[thread_1D_pos];
  redChannel[thread_1D_pos] = px.x;
  greenChannel[thread_1D_pos] = px.y;
  blueChannel[thread_1D_pos] = px.z;
}

__global__ void
recombine_blended_channels_within_interior(const float* const redChannel,
                                           const float* const greenChannel,
                                           const float* const blueChannel,
                                           uchar4* const outputImageRGBA,
                                           const size_t numRows,
                                           const size_t numCols,
                                           const unsigned char* const d_sourceMaskInteriorMap)
{
  const int2 thread_2D_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_id.x >= numCols || thread_2D_id.y >= numRows)
    return;
  const int thread_1D_id = thread_2D_id.y * numCols + thread_2D_id.x;
  if (d_sourceMaskInteriorMap[thread_1D_id] != 4)
    return;

  unsigned char red   = redChannel[thread_1D_id];
  unsigned char green = greenChannel[thread_1D_id];
  unsigned char blue  = blueChannel[thread_1D_id];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_id] = outputPixel;
}

__global__ void
copy_char_to_float(float* const large,
                   const unsigned char* const small,
                   const size_t numRows,
                   const size_t numCols)
{
  const int2 thread_2D_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_id.x >= numCols || thread_2D_id.y >= numRows)
    return;
  const int thread_1D_id = thread_2D_id.y * numCols + thread_2D_id.x;

  large[thread_1D_id] = (float)small[thread_1D_id];
}

__global__ void
recombine_channels(const float* const redChannel,
                   const float* const greenChannel,
                   const float* const blueChannel,
                   uchar4* const outputImageRGBA,
                   const size_t numRows,
                   const size_t numCols)
{
  const int2 thread_2D_id = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                                      blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_id.x >= numCols || thread_2D_id.y >= numRows)
    return;
  const int thread_1D_id = thread_2D_id.y * numCols + thread_2D_id.x;

  unsigned char red   = redChannel[thread_1D_id];
  unsigned char green = greenChannel[thread_1D_id];
  unsigned char blue  = blueChannel[thread_1D_id];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_id] = outputPixel;
}

uchar4* d_sourceImg;
unsigned char* d_sourceMask;

unsigned char* d_sourceMaskInteriorMap;

uchar4* d_targetImg;
unsigned char* d_targetRed;
unsigned char* d_targetGreen;
unsigned char* d_targetBlue;

unsigned char* d_sourceRed;
unsigned char* d_sourceGreen;
unsigned char* d_sourceBlue;

float* d_prevRed;
float* d_prevGreen;
float* d_prevBlue;
float* d_nextRed;
float* d_nextGreen;
float* d_nextBlue;

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{
  dim3 numThreads(32,32);
  dim3 numBlocks(1 + numColsSource / numThreads.x, 1 + numRowsSource / numThreads.y);
  const int img_size = sizeof(uchar4)*numColsSource*numRowsSource;
  const int chan_size = sizeof(unsigned char)*numColsSource*numRowsSource;


  // 1) Compute a mask of the pixels from the source image to be copied
     // The pixels that shouldn't be copied are completely white, they
     // have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
  checkCudaErrors(cudaMalloc(&d_sourceImg, img_size));
  checkCudaErrors(cudaMemcpy(d_sourceImg, h_sourceImg, img_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_sourceMask, chan_size));
  checkCudaErrors(cudaMemset(d_sourceMask, 0, chan_size));
  map_source_to_mask<<<numBlocks, numThreads>>>(d_sourceImg, d_sourceMask, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 2) Compute the interior and border regions of the mask.  An interior
  //    pixel has all 4 neighbors also inside the mask.  A border pixel is
  //    in the mask itself, but has at least one neighbor that isn't.
  checkCudaErrors(cudaMalloc(&d_sourceMaskInteriorMap, chan_size));
  checkCudaErrors(cudaMemset(d_sourceMaskInteriorMap, 0, chan_size));
  stencil_2d_von_neumann<<<numBlocks, numThreads>>>(d_sourceMask, d_sourceMaskInteriorMap, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 3) Separate out the incoming image into three separate channels
  checkCudaErrors(cudaMalloc(&d_sourceRed, chan_size));
  checkCudaErrors(cudaMalloc(&d_sourceGreen, chan_size));
  checkCudaErrors(cudaMalloc(&d_sourceBlue, chan_size));
  separateChannels<<<numBlocks, numThreads>>>(d_sourceImg, numRowsSource, numColsSource,
                                              d_sourceRed, d_sourceGreen, d_sourceBlue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 3.5) Separate out the target/destination image into three separate channels
  checkCudaErrors(cudaMalloc(&d_targetImg, img_size));
  checkCudaErrors(cudaMemcpy(d_targetImg, h_destImg, img_size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc(&d_targetRed, chan_size));
  checkCudaErrors(cudaMalloc(&d_targetGreen, chan_size));
  checkCudaErrors(cudaMalloc(&d_targetBlue, chan_size));
  separateChannels<<<numBlocks, numThreads>>>(d_targetImg, numRowsSource, numColsSource,
                                              d_targetRed, d_targetGreen, d_targetBlue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 4) Create two float(!) buffers for each color channel that will
  //    act as our guesses.  Initialize them to the respective color
  //    channel of the source image since that will act as our intial guess.
  const int size = sizeof(float)*numColsSource*numRowsSource;
  checkCudaErrors(cudaMalloc(&d_prevRed, size));
  checkCudaErrors(cudaMalloc(&d_prevGreen, size));
  checkCudaErrors(cudaMalloc(&d_prevBlue, size));
  checkCudaErrors(cudaMalloc(&d_nextRed, size));
  checkCudaErrors(cudaMalloc(&d_nextGreen, size));
  checkCudaErrors(cudaMalloc(&d_nextBlue, size));
  // Can't do memcpy since each data unit is of different size
  copy_char_to_float<<<numBlocks, numThreads>>>(d_prevRed, d_sourceRed, numRowsSource, numColsSource);
  copy_char_to_float<<<numBlocks, numThreads>>>(d_prevGreen, d_sourceGreen, numRowsSource, numColsSource);
  copy_char_to_float<<<numBlocks, numThreads>>>(d_prevBlue, d_sourceBlue, numRowsSource, numColsSource);
  copy_char_to_float<<<numBlocks, numThreads>>>(d_nextRed, d_sourceRed, numRowsSource, numColsSource);
  copy_char_to_float<<<numBlocks, numThreads>>>(d_nextGreen, d_sourceGreen, numRowsSource, numColsSource);
  copy_char_to_float<<<numBlocks, numThreads>>>(d_nextBlue, d_sourceBlue, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // 5) For each color channel perform the Jacobi iteration described
  //    above 800 times.
  for (int i = 0; i < JACOBI_ITR; i++) {
    if (i % 2 == 0) {
      poisson_equation_jacobi_iteration<<<numBlocks, numThreads>>>(
                                    d_nextRed, d_prevRed,
                                    d_sourceRed, d_targetRed,
                                    d_sourceMaskInteriorMap, d_sourceMask,
                                    numRowsSource, numColsSource);
      poisson_equation_jacobi_iteration<<<numBlocks, numThreads>>>(
                                    d_nextGreen, d_prevGreen,
                                    d_sourceGreen, d_targetGreen,
                                    d_sourceMaskInteriorMap, d_sourceMask,
                                    numRowsSource, numColsSource);
      poisson_equation_jacobi_iteration<<<numBlocks, numThreads>>>(
                                    d_nextBlue, d_prevBlue,
                                    d_sourceBlue, d_targetBlue,
                                    d_sourceMaskInteriorMap, d_sourceMask,
                                    numRowsSource, numColsSource);
    } else {
      poisson_equation_jacobi_iteration<<<numBlocks, numThreads>>>(
                                    d_prevRed, d_nextRed,
                                    d_sourceRed, d_targetRed,
                                    d_sourceMaskInteriorMap, d_sourceMask,
                                    numRowsSource, numColsSource);
      poisson_equation_jacobi_iteration<<<numBlocks, numThreads>>>(
                                    d_prevGreen, d_nextGreen,
                                    d_sourceGreen, d_targetGreen,
                                    d_sourceMaskInteriorMap, d_sourceMask,
                                    numRowsSource, numColsSource);
      poisson_equation_jacobi_iteration<<<numBlocks, numThreads>>>(
                                    d_prevBlue, d_nextBlue,
                                    d_sourceBlue, d_targetBlue,
                                    d_sourceMaskInteriorMap, d_sourceMask,
                                    numRowsSource, numColsSource);
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }

  // 6) Create the output image by replacing all the interior pixels
  //    in the destination image with the result of the Jacobi iterations.
  //    Just cast the floating point values to unsigned chars since we have
  //    already made sure to clamp them to the correct range.
  // recombine_blended_channels_within_interior<<<numBlocks, numThreads>>>(d_prevRed,
  //                                                                       d_prevGreen,
  //                                                                       d_prevBlue,
  //                                                                       d_targetImg,
  //                                                                       numRowsSource,
  //                                                                       numColsSource,
  //                                                                       d_sourceMaskInteriorMap);
  recombine_channels<<<numBlocks, numThreads>>>(d_sourceMaskInteriorMap,
                                                d_sourceMaskInteriorMap,
                                                d_sourceMaskInteriorMap,
                                                d_targetImg,
                                                numRowsSource,
                                                numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(h_blendedImg, d_targetImg, img_size, cudaMemcpyDeviceToHost));

  //  Since this is final assignment we provide little boilerplate code to
  //  help you.  Notice that all the input/output pointers are HOST pointers.

  //  You will have to allocate all of your own GPU memory and perform your own
  //  memcopies to get data in and out of the GPU memory.

  //  Remember to wrap all of your calls with checkCudaErrors() to catch any
  //  thing that might go wrong.  After each kernel call do:

  //  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //  to catch any errors that happened while executing the kernel.
}
