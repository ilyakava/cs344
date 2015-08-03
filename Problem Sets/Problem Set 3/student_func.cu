/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include <stdio.h>
#include "utils.h"

#define NUM_THREADS 1024

__global__
void shmem_min_max_reduce(const float * const d_in, float * d_out, const size_t numRows, const size_t numCols, const int maxMode)
{
  extern __shared__ float s_in[];

  int tid = threadIdx.x;
  int offset = blockIdx.x * blockDim.x + tid; // offset in global memory that is
  int imgSize = numRows * numCols;
  if (offset >= imgSize)
    return;

  s_in[tid] = d_in[offset];
  __syncthreads();

  // use right shift '>>' to divide by 2 each iteration
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && (tid + s) < imgSize)
      if (maxMode) {
        s_in[tid] = max(s_in[tid], s_in[tid + s]);
      } else {
        s_in[tid] = min(s_in[tid], s_in[tid + s]);
      }
    __syncthreads();
  }

  if (tid == 0) {
    *(d_out + blockIdx.x) = s_in[0];
  }
}

__global__
void global_histogram(const float* const d_logLuminance,
                      unsigned int* const d_out,
                      const float min_logLum,
                      const float range_logLum,
                      const size_t numRows,
                      const size_t numCols,
                      const size_t numBins)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if ( thread_2D_pos.y >= numRows || thread_2D_pos.x >= numCols )
    return;
  const unsigned int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  float ll = d_logLuminance[thread_1D_pos];
  int bin = (int)(((ll - min_logLum) / range_logLum) * numBins);

  // assert(min_logLum <= ll);
  // assert((min_logLum + range_logLum) >= ll);

  atomicAdd((d_out + bin), 1);
}

__global__
void hillis_steele_exclusive_scan(unsigned int* const d_pdf, const size_t numBins)
{
  int tid = threadIdx.x;
  // use left shift '<<' to multiply by 2 each iteration
  for (unsigned int s = 1; s < numBins; s <<= 1) {
    int left_neighborid = tid - s;
    if (left_neighborid >= 0)
      d_pdf[tid] = d_pdf[tid] + d_pdf[left_neighborid];
    __syncthreads();
  }

  // convert the above inclusive scan to an exclusive one
  if (threadIdx.x == 0)
    for (int i = (numBins-1); i > 0; i--)
      d_pdf[i] = d_pdf[i-1];
    d_pdf[0] = 0;
}

float *d_min_intermediate, *d_max_intermediate, *d_min_final, *d_max_final;

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  int blocks = 1 + ((numRows*numCols) / NUM_THREADS);

  // 1) find the minimum and maximum value in the input logLuminance channel
  //    store in min_logLum and max_logLum

  // two kernels, reduction to find min and max
  // min
  checkCudaErrors(cudaMalloc(&d_min_intermediate, sizeof(float) * NUM_THREADS));
  shmem_min_max_reduce<<<blocks, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(d_logLuminance, d_min_intermediate, numRows, numCols, 0);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMalloc(&d_min_final, sizeof(float)));
  shmem_min_max_reduce<<<1, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(d_min_intermediate, d_min_final, numRows, numCols, 0);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(&min_logLum, d_min_final, sizeof(float), cudaMemcpyDeviceToHost));
  // max
  checkCudaErrors(cudaMalloc(&d_max_intermediate, sizeof(float) * NUM_THREADS));
  shmem_min_max_reduce<<<blocks, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(d_logLuminance, d_max_intermediate, numRows, numCols, 1);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMalloc(&d_max_final, sizeof(float)));
  shmem_min_max_reduce<<<1, NUM_THREADS, NUM_THREADS * sizeof(float)>>>(d_max_intermediate, d_max_final, numRows, numCols, 1);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(&max_logLum, d_max_final, sizeof(float), cudaMemcpyDeviceToHost));

  // 2) subtract them to find the range
  printf("GPU min: %f\n", min_logLum);
  printf("GPU max: %f\n", max_logLum);
  float range_logLum = max_logLum - min_logLum;

  // 3) generate a histogram of all the values in the logLuminance channel using
  //    the formula: bin = (lum[i] - lumMin) / lumRange * numBins

  const dim3 blockSize(32,16,1);
  const dim3 gridSize(1 + numCols / blockSize.x, 1 + numRows / blockSize.y, 1);

  // first lets try with atomicAdd
  // later: give every thread local bins, then reduce
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  global_histogram<<<gridSize, blockSize>>>(d_logLuminance, d_cdf, min_logLum, range_logLum, numRows, numCols, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // debug
  // unsigned int bins[numBins];
  // checkCudaErrors(cudaMemcpy(bins, d_cdf, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToHost));
  // printf("PDF:\n");
  // for (int i = 0; i < numBins; i++)
  //   printf("%i,", bins[i]);

  // 4) Perform an exclusive scan (prefix sum) on the histogram to get
  //    the cumulative distribution of luminance values (this should go in the
  //    incoming d_cdf pointer which already has been allocated for you)

  // we have 1024 bins, and we have 2880 Thread processors on a K40c, and
  // 512 on a M2090, so we have plenty of workers, want step efficiency over
  // work efficiency, will use Hillis & Steele
  hillis_steele_exclusive_scan<<<1, numBins>>>(d_cdf, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // debug
  // checkCudaErrors(cudaMemcpy(bins, d_cdf, sizeof(unsigned int)*numBins, cudaMemcpyDeviceToHost));
  // printf("\nCDF:\n");
  // for (int i = 0; i < numBins; i++)
  //   printf("%i,", bins[i]);

}
