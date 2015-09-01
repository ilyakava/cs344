Optimizing for more perf

Arithmetic intensity: math/memory, want to maximize this. Example is to coalesce memory access: best way to minimize time spent on memory operations.

# Thinking about optimization

## Levels of optimization

1. Picking good algorithms
    - GPU: fundamentally parallel
        - merge over heap sort
2. Basic principles for efficiency
3. Arch specific detailed optimizations (Ninja: last few %)
    - Vector registers SSE, AVX
        - ignoring these gives you 1/4 to 1/8 max performance
    - Blocking for L1 cache (sizing working set appropriately)
    - Big difference between CPU and GPU programming, more CPU Ninja stuff might have 80%
    - GPU: We will touch upon: shared memory bank conflicts
4. Micro-optimization at instruction level
    - quake III hack: [recipSqrt 0x5f3759df](https://en.wikipedia.org/wiki/Fast_inverse_square_root)
    - GPU: floating point denorms (additional integer math performance)

First two are 3-10x. Last two are less than 1x.

## APOD - A Systematic Optimization Process/Cycle

- analyze
    - profiling (kernels and everything else)
        - how much could we benefit?
        - understand hotspots and move attention to bottlenecks
            - Amdahl's Law (1/(1-P)), P = parallelizable time
    - `gProf`, `VTune`, `VerySleepy`
- parallelize
    - "the best code is the code you don't have to write": Libraries
    - Directives (OpenMP, OpenACC on GPU)
    - pick an approach/algorithm
- optimize
    - cycle back and forth to parallelize
    - measure!
- deploy
    - "make it real"
- repeat

## Weak vs Strong scaling

Weak: Fold bigger protein, or more small ones, in same time
Strong: solve same problem faster


## Profiling

- Theoretical memory speed limit:
    - (Memory clock (Mhz) * Memory bus (bits)) / 8e3 = GB/s
        - if we achieve 40-60% okay
        - 60-75% good
        - >75% excellent (no optimization left)
- Calculating our code's memory speed:
    - With a pencil: (num elements read or written)*(bytes/element) / runtime
    - With a computer: nSight Eclipse/Visual Studio
        - nVidia product [nvvp](https://developer.nvidia.com/nvidia-visual-profiler)
        - Analysis tab: run analyze program
            - Global **Load** Efficiency
                - (of all bytes fetched, how many were useful?) - 100% for fully coalesced
                - `nvprof --metrics gld_efficiency`
                - if over 100% its an [estimation error](https://devtalk.nvidia.com/default/topic/628818/cuda-programming-and-performance/-ldquo-global-load-efficiency-rdquo-over-100-in-visual-profiler/post/3992785/#3992785)
            - Global **Store** Efficiency
                - `nvprof --metrics gst_efficiency`
            - DRAM utilization
                - `nvprof --metrics dram_utilization`
                - high = good
            - Shared Memory Replay Overhead
                - could be from shared memory bank conflicts (almost Ninja)
            - To see all `nvprof` metrics: `nvprof --devices 0 --query-metrics`

# Parallelize Example: Matrix transpose

1. serial 466ms
2. parallel per row 4.7ms (1.76% DRAM utilization)
3. parallelize in tiles 0.667 ms (16.1% DRAM utilization (12.5 GB/s))
  - no more parallelism to extract?
      - really 16.1% is pretty low
      - first step: Coalescing memory
          - minimize amount of memory transactions by minimizing distance of memory addresses that adjacent threads access
  - However, its not always optimal to do extract all the parallelism. Sometimes it pays to do more work per thread: "granularity coarsening"
4. Take advantage of shared memory to coalesce writes as well as reads (17.1%)
    - Tiling fix:
        - problem - coalesced reads, scattered writes (can't just flip these two)
        - goal - coalesce both (adjacent threadIdx.x writes to adjacent global memory)
        - solution - transpose in shared memory
5. Change tile size to fully load SMs (43.5%)
    - next: shared memory bank conflicts (subtle) to avoid replays
        - fix is to pad the shared memory and stripe it (45.0%)

Our Limits: Time spent fetching data, or ~~time spent computing~~

# How to make your code fast

- GPUs are fast b/c
    - massively parallel
    - extremely high-bandwidth memory (want to use all available memory bandwidth)

## Little's Law

- optimize customers in line at Starbucks coffee
- **number bytes delivered = avg latency per transaction * bandwidth**
    - *useful* bytes (why we want to coalesce, make every byte delivered be used)
- we want to maximize bandwidth
    - many bytes delivered, with low latency
    - 3 questions:
        1. are the bytes delivered all useful (coalesce)?
        2. are we suffering from too few transactions per thread?
        3. is our latency per transaction low?
    - (many servers serving multiple cups of coffee & servers work faster)

## Implications of Little's Law

- DRAM transaction: 100's cycles
    - thread accessing global mem must wait
    - many threads in flight
- DRAM = Pipe
    - **wide** and long (mem transaction takes a while, and many can go into the pipe)
    - take advantage of width
        - could use bigger words to deliver more adjacent data (make memory transactions wider)
            - useful but not vital ninja optimization

## Occupancy

- SM has limited resources (`deviceQuery`) (easy to take advantage of are in **bold**):
    1. 8 thread blocks
        - To calculate your usage of this: `Maximum number of threads per block:`
    2. **max threads runable on an SM** 1536/2048
        - View yours: `Maximum number of threads per multiprocessor`
    3. registers for all threads (65536)
        - View yours: `Total number of registers available per block`
        - [calculate your usage w/ compiler help](http://stackoverflow.com/a/17913801/2256243)
            - `nvcc -Xptxas -v`
        - Tweaking this is ninja level
    4. **bytes of shared memory** (16K-48K)
        - View yours: `Total amount of shared memory per block`
- Since all threads in a block must run on a single SM (Lesson 2), we want there to be no partial resources left over on an SM to maximize # thread blocks simultaneously runable
- Increasing Occupancy is a trade-off (usually with memory coalescing)
- CUDA Occupancy Calculator
    - `/home/user/cuda/NVIDIA_CUDA-7.0_Samples/0_Simple/simpleOccupancy/`

## Optimize Compute Performance

- minimize time at barriers (seen this with __syncthreads())
- minimize thread divergence
    - SIMD: single instruction multiple data
        - amortize code across data - save lots of power and transistors
        - on modern CPUs: SSE/AVX (affect more data (4x/8x) per operation)
    - SIMT: single instruction multiple threads
        - useful for divergence
        - what happens at a branch?

### Warp (group of threads)

- set of threads that execute same instruction at the same time
- will lead to some threads being paused
    - more divergence = more pausing
    - warps only have 32 threads, so 32x is the worst slowdown, can only switch 31 times
    - to understand the slowdown; think of 32 adjacent threads (in x direction), within those 32 how many groups of threads have different branches?
        - Cuda launches x, then y, then z ids
- in addition to if/else, loops:
    - think about how m any loops the avg warp will execute
- should count percentage of threads launched that are in diverged warps (and also degree of imbalance in thread workloads) to know if there is a point in reducing divergence
    - try to restructure, coarsely sort work items for size
- threads diverging outside of warps come with no additional penalty

## Latencies involved in different math ops (Half a ninja)

- use double precision only when you need it (use the `f` suffix in float ops)
- use intrinsics (built in, 2-3 bits less precision at massive speedup)

## Optimizations at system level

- PCIe (express bus) to communicate between CPU and GPU
- pinned host memory, staging area for transferring into GPU (extra copy into staging area takes time)
    - cudaHostMalloc()
    - cudaHostRegister() (to pin)
        - enables MemcpyAsync(); to return to Host (also introduces Streams)

# Streams

- sequence of operations that execute in order
    - memory transfers
    - kernels
- cudaMemcpyAsync() generically will block other cudaMemcpyAsync() calls. But placing them into different streams allows them to execute concurrently
    - cudaStream_t
- host launches the commands and returns immediately, activities will just pile up in different streams
- useful for huge amounts of data:
    - might want to copy data in chunks and process
        - limited by Host->Device memory transfers
    - streams allow asynchronous copying (overlap data transfer and computation)
- also helps fill gpu with many problems with limited parallelisms
- for more: check out streams and events is CUDA programming guide

# Problem set 5 - Faster Histogram

1. sort input data into coarse bins
2. Local histogram for local bins
3. concatenate coarse histograms (fits into shared memory)
