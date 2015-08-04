//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

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
  int predicate = (d_inputVals[id] & bit) == 0;
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
void exclusive_blelloch_scan(unsigned int* const d_list, const size_t numElems)
{
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;
  // reduce
  unsigned int i;
  for (i = 2; i <= numElems/2; i <<= 1) {
    if ((id + 1) % i == 0) {
      unsigned int neighbor_offset = i>>1;
      d_list[id] += d_list[id - neighbor_offset];
    }
    __syncthreads();
  }
  i >>= 1; // return i to last value before for loop exited
  // reset last to identity element
  if (id == (numElems-1))
    d_list[id] = 0;
  // downsweep
  for (i = i; i >= 2; i >>= 1) {
    if((id + 1) % i == 0) {
      unsigned int neighbor_offset = i>>1;
      unsigned int old_neighbor = d_list[id - neighbor_offset];
      d_list[id - neighbor_offset] = d_list[id]; // copy
      d_list[id] += old_neighbor;
    }
    __syncthreads();
  }
}

__global__
void scatter(unsigned int* const d_input, unsigned int* const d_output,
             unsigned int* const d_predicateTrueScan, unsigned int* const d_predicateFalseScan,
             unsigned int* const d_predicateFalse, unsigned int numPredicateTrueElements,
             const size_t numElems)
{
  const unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id >= numElems)
    return;

  unsigned int newLoc;
  if (d_predicateFalse == (unsigned int)1)
    newLoc = d_predicateFalseScan[id] + numPredicateTrueElements;
  else
    newLoc = d_predicateTrueScan[id];
  d_output[newLoc] = d_input[id];
}

unsigned int* const d_predicate, d_predicateTrueScan, d_predicateFalseScan, ping, pong;

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  size_t size = sizeof(unsigned int) * numElems;
  int blockSize = 1024;
  int gridSize = 1 + blockSize / numElems;

  checkCudaErrors(cudaMalloc(&d_predicate, size));
  checkCudaErrors(cudaMalloc(&d_predicateTrueScan, size));
  checkCudaErrors(cudaMalloc(&d_predicateFalseScan, size));

  unsigned int lastPredicateTrue, numPredicateTrueElements, nsb;

  for (unsigned int bit = 0; bit <= 32; bit++) {
    nsb = 1<<bit;
    // create predicateTrue
    check_bit<<<gridSize, blockSize>>>(d_inputVals, d_predicate, nsb, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // scan predicateTrue
    checkCudaErrors(cudaMemcpy(&d_predicateTrueScan, d_predicate, size, cudaMemcpyDeviceToDevice));
    exclusive_blelloch_scan<<<gridSize, blockSize>>>(d_predicateTrueScan, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // determine offset of 2nd bin, i.e. how many items are in the 1st bin,
    // i.e. for how many the predicate is TRUE
    checkCudaErrors(cudaMemcpy(&lastPredicateTrue, (d_predicate + numElems - 1),
                               sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&numPredicateTrueElements, (d_predicateTrueScan + numElems - 1),
                               sizeof(unsigned int), cudaMemcpyDeviceToHost));
    numPredicateTrueElements += lastPredicateTrue;
    // transform predicateTrue -> predicateFalse
    flip_bit<<<gridSize, blockSize>>>(d_predicate, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // scan predicateFalse
    checkCudaErrors(cudaMemcpy(&d_predicateFalseScan, d_predicate, size, cudaMemcpyDeviceToDevice));
    exclusive_blelloch_scan<<<gridSize, blockSize>>>(d_predicateFalseScan, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // scatter values (flip input/output depending on iteration)
    if ((bit + 1) % 2 == 1) {
      ping = &d_inputVals;
      pong = &d_outputVals;
    } else {
      ping = &d_outputVals;
      pong = &d_inputVals;
    }
    scatter<<<gridSize, blockSize>>>(ping, pong, d_predicateTrueScan, d_predicateFalseScan,
                                     d_predicate, numPredicateTrueElements, numElems);
  }
}
