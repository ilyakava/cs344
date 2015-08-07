/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/


#include "utils.h"

__global__
void baseline(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numBins,
               const unsigned int numElems)
{
  // 1.4254ms
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  int bin = vals[id];
  atomicAdd((histo + bin), 1);
}

__global__
void distribute_atomics_on_shmem_first(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numBins,
               const unsigned int numElems)
{
  extern __shared__ unsigned int s_histo[];

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  unsigned int bin = vals[id];
  assert(bin < numBins);

  atomicAdd(&s_histo[bin], 1);

  __syncthreads();

  atomicAdd(&histo[threadIdx.x], 1);
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  int numThreads = 1024;
  int numBlocks = 1 + numElems / 1024;

  // baseline<<<numBlocks, numThreads>>>(d_vals, d_histo, numBins, numElems);
  distribute_atomics_on_shmem_first<<<numBlocks, numThreads, sizeof(unsigned int)*numBins>>>(d_vals, d_histo, numBins, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
