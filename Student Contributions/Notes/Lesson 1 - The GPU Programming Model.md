Clock speed increase means increase power consumption, a limit that we are at.
CPU instruction level parallelism is at a peak too.
Many smaller weaker processors

Arithmetic units (ALUs)
GPUs have more active pieces of work active that can be active (threads)

GPGPU (general programming on GPU)

Transistors get smaller each year, feature size (minimum transistor)
transistor size decreases, processor designers increase clock speeds,
last decade clock speeds are constant

Running many transistors generates a lot of heat, can't build processors in the same way (increasing speed of one) - Power is the biggest concern, instead, make smaller weaker more efficient processors, and have more of them

CPU not energy efficient b/c of control hardware (better performance), high power. For a fixed amount of power, you can get more bang for your buck with simpler control structures for many small processors.

Latency vs Throughput - trade off when building a better processor

Optimizing for throughput - long lines means there is never a time a worker doesn't have work waiting (and so is always working)

CPUs minimize latency (little time on each task elapsed)

More restrictive programming as a result of simpler control structure.

AVX vector operations (parallelism on CPUs)

---

CPU = Host, GPU = Device

Cuda programming model - both processors with one program. Cuda supports many languages, but we use C.

Assumes device is a co-processor to host, and that they have separate dedicated memories. CPU is in charge of main program

- moving CPU memory to GPU and back
- allocate GPU memory
- invoke programs (kernel), host launches kernels on the device

CPU is boss, GPU can only respond to requests. (newer GPUs can launch their own kernels - new thing)

High ratio of computation to communication (a lot of work on a little data)

Key idea: write a program for one thread and then launch 100k of them. Don't mention the level of parallelism in your program.

GPU good at:
- efficiently launching a large number of threads
- running lots of threads in parallel

Launch a kernel on 64 threads, each instance squares a number.
Each thread knows which thread it is (thread index), can assign nth thread to work on nth element of the array.

compiling:
`nvcc -o square square.cu`

`d_` and `h_` are variable names for device and host. Common beginner mistake for pointers.

To tell the computer where the variable is, use something like:
`((void **) &d_in, ARRAY_BYTES);`
to allocate, and then:
`cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpy???)`

Cuda launch `<<< ___ >>>` can only call on GPU data

declaration specifier, way that Cuda knows that its not CPU code `threadIdx` struct.

When launching a kernel, you specify the number of blocks and the number of threads per block. Many blocks at the same time, each block maxes out on either 512 or 1024 threads (new/old).

Each thread knows its index in its thread, and in its block.

Multiple dimensional layouts of blocks and threads are possible (1,2,3D for both grid of blocks, and grids of threads). (Blocks in y dimension, threads in x dimension).

GPUs are good at MAP, b/c of large number of processors, optimize for throughput.

# Problem Set 1 - Convert to Greyscale
