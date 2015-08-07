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

#define NUM_SHARED_HISTS 4
#define MAX_THREADS_PER_BLOCK 1024

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
  // 1.2942ms
  extern __shared__ unsigned int s_histo[];

  s_histo[threadIdx.x] = 0;
  __syncthreads();

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  unsigned int bin = vals[id];

  atomicAdd(&s_histo[bin], 1);

  __syncthreads();

  atomicAdd(&histo[threadIdx.x], s_histo[threadIdx.x]);
}

__global__
void reduce_on_shmem_first(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numBins,
               const unsigned int numElems)
{
  // basic var and bounds checking
  __shared__ unsigned int s_val[MAX_THREADS_PER_BLOCK];
  int tid = threadIdx.x;
  int binId = blockIdx.y;
  int id = blockDim.x * blockIdx.x + tid;

  // put initial values into shared histograms
  s_val[tid] = 0;
  if (id < numElems && (binId == vals[id])) {
    s_val[tid] = 1;
  }
  __syncthreads();

  // reduce
  for (unsigned int ithVal = 2; ithVal <= blockDim.x; ithVal <<= 1)
  {
    unsigned int neighborOffset = ithVal>>1;
    if (((tid + 1) % ithVal) == 0) {
      s_val[tid] += s_val[tid - neighborOffset];
    }
    __syncthreads();
  }

  // write output
  if (tid == (blockDim.x - 1)) {
    atomicAdd(&histo[binId], s_val[tid]);
  }
}

void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  // int numThreads = NUM_SHARED_HISTS;

  const dim3 numThreads(MAX_THREADS_PER_BLOCK, 1, 1);
  const dim3 numBlocks(1 + numElems / numThreads.x, numBins, 1);

  // baseline<<<numBlocks, numThreads>>>(d_vals, d_histo, numBins, numElems);
  // distribute_atomics_on_shmem_first<<<numBlocks, numThreads, sizeof(unsigned int)*numThreads>>>(d_vals, d_histo, numBins, numElems);
  reduce_on_shmem_first<<<numBlocks, numThreads, sizeof(unsigned int)*MAX_THREADS_PER_BLOCK>>>(d_vals, d_histo, numBins, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
