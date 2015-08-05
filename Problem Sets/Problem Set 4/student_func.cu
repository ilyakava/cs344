//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <math.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

__global__
void check_bit(unsigned int* const d_inputVals, unsigned int* const d_outputPredicate,
               const unsigned int bit, const size_t numElems)
{
  // this predicate returns TRUE when the significant bit is not present
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  int predicate = ((d_inputVals[id] & bit) == 0);
  d_outputPredicate[id] = predicate;
}

__global__
void flip_bit(unsigned int* const d_list, const size_t numElems)
{
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  d_list[id] = ((d_list[id] + 1) % 2);
}

__global__
void partial_exclusive_blelloch_scan(unsigned int* const d_list, unsigned int* const d_block_sums, const size_t numElems)
{
  extern __shared__ unsigned int s_block_scan[];

  const unsigned int tid = threadIdx.x;
  const unsigned int id = blockDim.x * blockIdx.x + tid;

  // copy to shared memory, pad the block that is too small
  if (id >= numElems)
    s_block_scan[tid] = 0;
  else
    s_block_scan[tid] = d_list[id];
  __syncthreads();
  printf("in kernel (thread: %i, block: %i of %i): %i %i %i %i %i %i %i %i\n",tid, blockIdx.x, blockDim.x, s_block_scan[0], s_block_scan[1], s_block_scan[2], s_block_scan[3], s_block_scan[4], s_block_scan[5], s_block_scan[6], s_block_scan[7]);

  // reduce
  unsigned int i;
  for (i = 2; i <= blockDim.x; i <<= 1) {
    if ((tid + 1) % i == 0) {
      unsigned int neighbor_offset = i>>1;
      s_block_scan[tid] += s_block_scan[tid - neighbor_offset];
    }
    __syncthreads();
  }
  i >>= 1; // return i to last value before for loop exited
  // reset last (sum of whole block) to identity element
  if (tid == (blockDim.x-1))
    d_block_sums[blockIdx.x] = s_block_scan[tid];
    s_block_scan[tid] = 0;
  __syncthreads();

  // printf("in kernel (thread: %i, block: %i of %i): %i %i %i %i %i %i %i %i\n",tid, blockIdx.x, blockDim.x, s_block_scan[0], s_block_scan[1], s_block_scan[2], s_block_scan[3], s_block_scan[4], s_block_scan[5], s_block_scan[6], s_block_scan[7]);

  // downsweep
  for (i = i; i >= 2; i >>= 1) {
    if((tid + 1) % i == 0) {
      unsigned int neighbor_offset = i>>1;
      unsigned int old_neighbor = s_block_scan[tid - neighbor_offset];
      s_block_scan[tid - neighbor_offset] = s_block_scan[tid]; // copy
      s_block_scan[tid] += old_neighbor;
    }
    __syncthreads();
  }

  // copy result to global memory
  if (id < numElems)
    d_list[id] = s_block_scan[tid];
}

__global__
void increment_blelloch_scan_with_block_sums(unsigned int* const d_predicateScan,
                                             unsigned int* const d_block_sums, const size_t numElems)
{
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  d_predicateScan[id] += d_block_sums[blockIdx.x];
}

__global__
void scatter(unsigned int* const d_input, unsigned int* const d_output,
             unsigned int* const d_predicateTrueScan, unsigned int* const d_predicateFalseScan,
             unsigned int* const d_predicateFalse, unsigned int* const d_numPredicateTrueElements,
             const size_t numElems)
{
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  unsigned int newLoc;
  if (d_predicateFalse[id] == 1) {
    newLoc = d_predicateFalseScan[id] + *d_numPredicateTrueElements;
  } else {
    newLoc = d_predicateTrueScan[id];
  }


  if (newLoc >= numElems)
    printf("ALERT d_predicateFalse[id]: %i newLoc: %i numElems: %i\n", d_predicateFalse[id], newLoc, numElems);


  d_output[newLoc] = d_input[id];
}

unsigned int* d_predicate;
unsigned int* d_predicateTrueScan;
unsigned int* d_predicateFalseScan;
unsigned int* d_numPredicateTrueElements;
unsigned int* d_numPredicateFalseElements;
unsigned int* d_block_sums;

// DEBUG
void print_array(unsigned int* array, size_t length)
{
  for (int i = 0; i < length; i++)
    printf("%i ", array[i]);
  printf("\n");
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  printf("numElems: %i\n", numElems);


  // DEBUG
  size_t myNumElems = 8;
  int blockSize = 8;
  printf("myNumElems: %i\n", myNumElems);


  size_t size = sizeof(unsigned int) * myNumElems;
  int gridSize = ceil(myNumElems / blockSize);

  unsigned int h_array[myNumElems];

  unsigned int h_predicateTrue[myNumElems];
  unsigned int h_predicateTrueScan[myNumElems];
  unsigned int nsb;
  unsigned int* h_numPredicateTrueElements = (unsigned int *)malloc(sizeof(unsigned int));

  checkCudaErrors(cudaMalloc((void**)&d_predicate, size));
  checkCudaErrors(cudaMalloc((void**)&d_predicateTrueScan, size));
  checkCudaErrors(cudaMalloc((void**)&d_predicateFalseScan, size));
  checkCudaErrors(cudaMalloc((void**)&d_numPredicateTrueElements, sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc((void**)&d_numPredicateFalseElements, sizeof(unsigned int))); // throwaway
  checkCudaErrors(cudaMalloc((void**)&d_block_sums, gridSize*sizeof(unsigned int)));

  unsigned int max_bits = 1;
  for (unsigned int bit = 0; bit < max_bits; bit++) {


    // DEBUG
    // checkCudaErrors(cudaMemcpy(&h_array, d_inputVals, size, cudaMemcpyDeviceToHost));
    // printf("bit %i array:\n", bit);
    // print_array(h_array, myNumElems);


    nsb = 1<<bit;
    // create predicateTrue
    if ((bit + 1) % 2 == 1) {
      check_bit<<<gridSize, blockSize>>>(d_inputVals, d_predicate, nsb, myNumElems);
    } else {
      check_bit<<<gridSize, blockSize>>>(d_outputVals, d_predicate, nsb, myNumElems);
    }
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // scan predicateTrue
    printf("SCAN PREDICATE TRUE:\n");
    checkCudaErrors(cudaMemcpy(d_predicateTrueScan, d_predicate, size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(d_block_sums, 0, gridSize*sizeof(unsigned int)));
    partial_exclusive_blelloch_scan<<<gridSize, blockSize, sizeof(unsigned int)*blockSize>>>(d_predicateTrueScan, d_block_sums, myNumElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    partial_exclusive_blelloch_scan<<<1, blockSize, sizeof(unsigned int)*blockSize>>>(d_block_sums, d_numPredicateTrueElements, gridSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    increment_blelloch_scan_with_block_sums<<<gridSize, blockSize>>>(d_predicateTrueScan, d_block_sums, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // determine offset of 2nd bin, i.e. how many items are in the 1st bin,
    // i.e. for how many the predicate is TRUE
    checkCudaErrors(cudaMemcpy(&h_predicateTrue, d_predicate,
                               size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&h_predicateTrueScan, d_predicateTrueScan,
                               size, cudaMemcpyDeviceToHost));
    *h_numPredicateTrueElements = h_predicateTrueScan[myNumElems-1] + h_predicateTrue[myNumElems-1];
    // checkCudaErrors(cudaMemcpy(d_numPredicateTrueElements, h_numPredicateTrueElements,
    //                            sizeof(unsigned int), cudaMemcpyHostToDevice));


    // DEBUG
    printf("h_predicateTrue:\n");
    print_array(h_predicateTrue, myNumElems);
    printf("h_predicateTrueScan:\n");
    print_array(h_predicateTrueScan, myNumElems);
    printf("h_numPredicateTrueElements: %i\n", *h_numPredicateTrueElements);


    // transform predicateTrue -> predicateFalse
    flip_bit<<<gridSize, blockSize>>>(d_predicate, myNumElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // scan predicateFalse
    printf("SCAN PREDICATE FALSE:\n");
    checkCudaErrors(cudaMemcpy(d_predicateFalseScan, d_predicate, size, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(d_block_sums, 0, gridSize*sizeof(unsigned int)));
    partial_exclusive_blelloch_scan<<<gridSize, blockSize, sizeof(unsigned int)*blockSize>>>(d_predicateFalseScan, d_block_sums, myNumElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    partial_exclusive_blelloch_scan<<<1, blockSize, sizeof(unsigned int)*blockSize>>>(d_block_sums, d_numPredicateFalseElements, gridSize);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    increment_blelloch_scan_with_block_sums<<<gridSize, blockSize>>>(d_predicateFalseScan, d_block_sums, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // scatter values (flip input/output depending on iteration)
    if ((bit + 1) % 2 == 1) {
      scatter<<<gridSize, blockSize>>>(d_inputVals, d_outputVals, d_predicateTrueScan, d_predicateFalseScan,
                                       d_predicate, d_numPredicateTrueElements, myNumElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      // scatter<<<gridSize, blockSize>>>(d_inputPos, d_outputPos, d_predicateTrueScan, d_predicateFalseScan,
      //                                  d_predicate, d_numPredicateTrueElements, myNumElems);
      // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    } else {
      scatter<<<gridSize, blockSize>>>(d_outputVals, d_inputVals, d_predicateTrueScan, d_predicateFalseScan,
                                       d_predicate, d_numPredicateTrueElements, myNumElems);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      // scatter<<<gridSize, blockSize>>>(d_outputPos, d_inputPos, d_predicateTrueScan, d_predicateFalseScan,
      //                                  d_predicate, d_numPredicateTrueElements, myNumElems);
      // cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    }
  }

  // DEBUG
  checkCudaErrors(cudaMemcpy(&h_array, d_inputVals, size, cudaMemcpyDeviceToHost));
  // printf("final array:\n");
  // print_array(h_array, myNumElems);

  int acc= 0;
  for (int i = 1; i < myNumElems; i++)
    if (h_array[i] > h_array[i-1])
      acc++;
  if (acc == (myNumElems-1))
    printf("SUCCESS CORRECT SORT\n");
  else
    printf("ERROR INCORRECT SORT\n");

  checkCudaErrors(cudaFree(d_predicate));
  checkCudaErrors(cudaFree(d_predicateTrueScan));
  checkCudaErrors(cudaFree(d_predicateFalseScan));
  checkCudaErrors(cudaFree(d_numPredicateTrueElements));
}
