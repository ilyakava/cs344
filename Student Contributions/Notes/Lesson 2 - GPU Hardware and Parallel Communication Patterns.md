Many threads working together. All about Communication:
- threads reading/writing to same piece of memory
- threads exchanging partial results

# Parallel Communication Patterns

1. Map: 1-to-1, simple correspondence
2. Gather: read from multiple, write single output (many-to-one)
3. Scatter read from single, increment in various outputs (piece of computation is scattered in memory) (one-to-many)
4. Stencil (no guards, writes in every location) (several-to-one, specialized gather operation)
    - Data reuse
    - Ex (+ symbol 2D von Neumann pattern)
    - Ex (3x3 block 2D Moore pattern)
5. Transpose: reorder elements in memory (also 1-to-1)
    - AoS Array of structures: ijijijijij
    - SoA Structure of arrays: iiiiijjjjj

---

6. reduce (all-to-one)
7. scan/sort (all-to-all)

# GPU hardware will affect our programming model

How to exploit data re-use
Safety

thread (path of execution (different threads take different paths)), kernel runs in many threads.

Thread blocks - group of threads that cooperate to solve a problem. Many blocks might run a kernel

SMs - streaming multiprocessors (1 = small, 16 = big GPU)
SM has memory and simple processors
GPU is responsible for allocating blocks to SMs

CUDA makes few guarantees when or where a block runs. Advantage is program is independent of GPU scale.

Dead lock - block x is waiting for another block y to give it a result, but block y already exited.

Means that threads and blocks must complete.

## Guarantees

- all the threads in the block are guaranteed to run on the same SM at the same time
- all blocks in a kernel finish before any blocks from next kernel run

Memory model
thread - local memory
thread block - shared memory between threads
every thread from any block - global memory

## Synchronization

Barrier - all threads stop here (`__syncthreads()`), happens between kernels

CUDA = hierarchy of computation, memory spaces, synchronization

# Writing Efficient Programs

## High Arithmetic intensity

arithmetic intensity = math/memory

- max compute ops/thread
- min time spent on mem/thread (no total mem ops, not tot mem, but time)
    - move freq accessed data to fast memory
    - local > shared >> global >> host (speed)

### Denominator: memory

#### Use Faster memory

- local memory is just a local var in the kernel
- global memory is when you declare a pointer on host, cudaMalloc with that pointer, and then pass the pointer to a kernel.
- shared memory: `__shared__` prefix

not all threads in a block execute at the same time, no guarantees here.

#### Coalesce

Another important thing for efficiency: coalesce access to global memory - threads are fastest when they read/write continuous global memory access. Called 'coalesced access' when you do so, otherwise could be called 'strided'. Even worse is 'random' which could lead to every thread having its own memory transaction.

#### Sometimes Atomic operations can slow things down

`g[i] = g[i] + 1`: read modify write operation
Doing this 10,000 times in parallel, will only not-collide ~450 times! Need atomic operations (B.12 in CUDA toolkit docs) like `atomicAdd` (which uses special built in hardware)

Limitations
- not everything is supported functions (like mod or exp) and datatypes (usually only ints) alike
    - plus side is you can implement anything by just using CAS (compare and swap) operation (example in docs)
- still no ordering constraints
    - floating point arithmetic non-associative
- serializes access to memory = slow!


| threads,elements,atomic? | audacity time (ms) | correct answer? | 40c time (us) | my order | udacity order |
|--------------------------|--------------------|-----------------|---------------|----------|---------------|
| 1M,1M,NA                 | .11                | y               | 64.8          | 2        | 2             |
| 1M,1M,A                  | .17                | y               | 43.9          | 1        | 4             |
| 1M,100,NA                | .32                | n               | 564           | 4        | 1             |
| 1M,100,A                 | .37                | y               | 70            | 3        | 3             |
| 10M,100,A                | 3.43               | y               | 682           | 5        | 5             |


## Avoid thread divergence

- different if/else control flow
- loops dependent on threadIdx

# Problem Set 2 - Blur

Will use stencil op (very common) for local averaging

- write blur kernel
- write kernel that separates r,g,b
- allocate memory for filter
- set grid and block size

## Organizing Parallel Threads

Problem with 1D grid with 1D blocks: `invalid configuration argument cudaGetLastError()` with a 5184x3456 image (`~/code/fundus/data/train/orig/6165_right.jpeg`)

Move to 2D grid with 2D blocks (consider a matrix as a *grid* of *block* matrices, and each block matrix is a *thread* matrix) for scalability.

| blockSize | time (ms) |
|-----------|-----------|
| 16,16,1   | 1.73      |
| 16,32,1   | 1.81      |
| 32,16,1   | 1.55      |
| 32,32,1   | 1.62      |
