# Fundamental GPU Algorithms (Reduce, Scan, Histogram)

Exploit concurrency.
GPU is well suited for independent operations, like the map operation, and stencil (more generally: gather).
All to 1, 1-to-many, more complicated get overlap. (Reduce, Scan, Histogram) are 3 primitives to implement things like this.

- Ideal scaling: adding 4x workers leads you to finish 4x sooner.
- Types of complexity
    - Step: how long it took to perform 1 unit of computation.
        - in graph example: step is each layer in the graph of nodes
    - Work: total amount of work, over all active workers, that it took to finish computation.
        - in graph example: total number of nodes
    - will compare these to serial implementations
    - and will usually talk of these as functions of size of input
    - parallel algorithm (P) is work efficient if work complexity is asymptotically the same, i.e. within a constant factor, as work complexity of sequential algorithm (S)
    - Reducing number of steps in P compared to S, leads to more efficient implementation

## Reduce

Interesting b/c requires cooperation, has dependencies.
We'll require 2 things of our reducing function:
- binary
- associative

- Serially: O(n) work and step complexity.
- Parallel: O(logn) step complexity (same work complexity)
    - this means that N=1024 will have 10 steps on a GPU, and be 2 orders of magnitude faster if the GPU can do all the work in a single step in parallel

If you are limited in number of processors: Brent's theorum
- A WT algorithm with step complexity S(n) and work complexity W(n) can be simulated on a p-processor PRAM in no more than ⌊ W(n)/p ⌋ + S(n) parallel steps.
- ([pg 5](http://cgvr.cs.uni-bremen.de/teaching/mpar_literatur/PRAM%20Algorithms%20-%20Chatterjee,%202009.pdf))

Shared data alocation: `extern __shared__ float sdata[];` and then passed as 3rd arg to launched kernel.

Shared memory can save us global memory bandwidth (calculate by summing all reads and writes together of global version, and dividing by the same for shared version). If this is 3x though, the only way you get 3x speed improvement is by "saturating the memory  system."
- Multiple items per thread
- warps are synchrous

## Scan

- like balancing your checkbook, a running sum
    - input is transactions (+/-) array
    - output is running sum array
    - 2 flavors:
        - exclusive: don't include element you are currently on
        - inclusive: do include
- not interesting in the serial world
- very useful
    - compaction
    - allocation
    - histogram
    - quicksort for sparse matrix computation and data compression
- inputs
    - array
    - operator
    - identity element
- complexity
    - serial
        - work: n
        - steps: n
    - parallel
        - lets pretend we'll do n reductions:
            - steps: same as reduction for largest input n: logn
            - work: n^2
        - Hillis & Steele
            - more step efficient (SE), first on GPUs
            - on step i, add yourself to your 2^i left neighbor
                - doubling the size of the communication on every step
            - work: nlogn
            - step: logn
        - Blelloch
            - more work efficient (WE)
            - reduce & downsweep
            - work: 2*n (same as serial ==  good!)
            - step: 2*logn

## When to choose WE over SE?

- WE good for more work than processors
- SE good for more processors than work (i.e. work doesn't matter)

## Histogram

- an exclusive scanning (after summing up counts w/in ranges)
- 3 ways to get around the race-condition of incrementing bin count
    - atomics (serialize access), bad for when you have few bins
        - basically limiting number of parallelisms, and limiting scalability as you get a great GPU
    - give every thread local bins, then reduce
        - local histograms using fastest memory
    - sort & reduce by key
        - k: v -> bin # -> count
        - sort ascending by key, reduce within a key (advantage of contiguous memory)
- plus combination of above

## Tone Mapping (HW)

- process of mapping image of wide range of brightness values to narow range
- Histogram equalization
    - map
    - reduce
    - scatter
    - scan
- Main idea:
    - get histogram of brightness values
        - find min/max with parallel reduce
        - use atomicAdd (at first)
    - scan histogram

- Questions:
    - how much shared memory are we limited to?