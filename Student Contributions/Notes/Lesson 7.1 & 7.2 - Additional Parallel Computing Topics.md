Broad Topics:

- Optimization strategies
- Libraries
- Programming power tools (STL or Boost for C++)
- Platforms (FORTRAN, Python, Matlab) and cross platform (opencl, openacc, opengl compute)
- Dynamic parallelism

Parallel Optimization

- Straton's Taxonomy. InPar2012, IEEE Computer (shorter)
- 7 basic techniques for GPU code
    1. Data layout transformation
        - coalescing is important b/c DRAM systems transfer large chunks/bursts of data per transaction
            - burst utilization (AoS -> SoA)
        - partition camping (Ninja level)
        - Array of Structures of tiled arrays (ASTA) (Ninja level)
    2. Scatter-to-gather
        - gather = overlapping reads are fine
        - scatter = many potential conflicts
        - binning
    3. Tiling
        - buffering repeated data reads into fast on-chip storage
        - CPU core's on chip cache, implicit copy
            - "cache blocking," not so great on GPU since there's not much memory per thread
        - scratchpad (shared memory on GPUs)
    4. Privatization
        - duplicate a piece of data for multiple threads, do shared computation, so that repeated writes to a single location are minimized
        - ex: (small) histogram
            - per thread privatized histograms, per block privatized histograms, the acc into global
            - for large (100s of bins), skip per thread privatization
    5. Binning / spatial data structures
        - optimize away redundant work of checking which (small subset of) threads will modify some output
    6. Compaction
        - know which exact elements that require computation
        - do extra computation if most require computation
        - use a dense data structure if few elements need computation
    7. Regularization
        - reorganize input data to reduce load imbalance
- As 7 questions for optimizing your code:
    1. Can we use a data structure that allows for more coalescing?
    2. Can we transform overlapping writes (scatter) to overlapping reads (gather)?
    3. Can we tile to share input for repeated reads?
    4. Can we Privatize to avoid sharing output?
    5. Can we reduce the amount of input necessary to look at for a given output?
    6. Can we avoid accessing inactive inputs?
    7. Can we extract regular parallelism from irregular parallelism?

# Libraries

- cuBLAS
- cuFFT (similar usage to FFTW)
- cuSPARSE (includes higher level routines like: incomplete LEU factorization)
- cuRAND
- NPP (Nvidia Performance Primitives) for image processing
- Magma GPU + CPU implementations of many LAPACK routines
- CULA (eigen solvers, matrix factorizations, there's also a sparse library for this too)
- ArrayFire data-parallel array manipulation (in between Programming power tool and domain programming library)

Using Libraries:

1. Substitute library call names
2. Manage data locality (host vs device)
3. Rebuild and link

# Power tools

- Less about solving a particular domain of problems and more about having programmers design their own solution
- Thrust
    - STL (useful abstractions for containers and iterators)
    - Also comes with niceties of the Boost library
    - Makes it easy for host code to manipulate data on GPU
        - `thrust::device_vector` generic containers than can resize dynamically
    - inter-operate with CUDA code easily by passing pointers
    - avoids (cut&paste, error prone) boilerplate code
- [CUB](http://nvlabs.github.io/cub/index.html) (Cuda Unbound)
    - Software reuse in CUDA kernels
    - call someone else's implementation of sort and can specify:
        - how many threads
        - how much shared memory
        - which algorithm?
    - compile time binding
    - enables optimization, autotuning
    - Accessing global memory can be complicated. CUB puts an abstraction around algorithm and memory access pattern
        - coalescing, Little's Law
        - Super ex: Kepler LDG intrinsic (Ninja level)
- CudaDMA
    - tackles a subset of what CUB tackles (CUB gets the computation too)
    - like CUB but specifically for global<->shared mem transactions
    - objects, explicit transfer pattern
        - sequential
        - strided
        - indirect (sparse matrix)
    - this decoupling helps programmability, portability, performance

# Other Platforms

- PyCuda (wraps)
- target CUDA directly
    - Copperhead
        - data parallel subset of python
        - generates thrust code
        - inter-operates with python code
    - CUDA Fortran
    - Halide
        - image processing DSL
    - Matlab
        - also can wrap
        - can call matlab on gpu data (`gpuArray`)
        - or can load a custom kernel

# Cross Platform Solutions

AMD, Intel, Arm

- similar to cuda:
    - OpenCL
    - OpenGL Compute
        - tightly integrated with OpenGL Graphics library
    - shares ideas of thread blocks
- Directives
    - OpenACC (compilers exist for C, C++, Fortran)
    - can transform a host for loop into GPU code (good for legacy code)
    - update over OpenMP

# Dynamic Parallelism (7.2)

Steven Jones

- Senior engineer on CUDA group
- works on CUDA programming model

- So far: in co-processor model (GPU controlled by CPU) GPU does not launch tasks for the GPU
- Bulk parallelism
    - everybody operates at the same time without dependencies
    - Ex: reduction
    - task parallelism: similar operation on large number of input (not just do one thing that fills GPU, but have a dozen independent things that make progress autonomously)
- Nested parallelism
    - parallel program creating other parallel program
    - Ex: Normalizing audio
        - find max volume in chunks
        - reduce volume in some areas
    - recursion
- Composability
    - kernels in a stream are interdependent
    - a parent waits for child sequence to finish
- Things to watch out for (since children are their own blocks)
    1. remember that many threads run the same program, don't launch more children than necessary by accident
    2. CUDA objects (streams and events) are private to a block, so you can only synchronize within a block
    3. Cannot pass shared memory to a child (need to use global)
- 