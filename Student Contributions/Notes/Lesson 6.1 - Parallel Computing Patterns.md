Some lessons to help you think in Parallel

# Dense N-body problem

Examples:

- gravitational interaction in space
- electrostatic force
- molecular dynamics (protein folding)

## All-pairs N-body. [Nyland, Harris, Prins](http://http.developer.nvidia.com/GPUGems3/gpugems3_ch31.html)

- brute force
    - simple, and we'll look at how to take advantage of memory hierarchy
- 2 steps usually
    - compute forces on each particle
    - move 1 step in simulation and repeat
- approximate (lump nearby points together)
    - tree methods (barnes hut)
    - fast multipole
- problem: brute force requires fetching each element 2N-1 times. we'd like to:
    - store and reuse memory fetches
    - take advantage to share data between threads
- solution 1: tile: PxP data, PxP threads
    - gets us a lot of parallel computation, with few fetches from global memory
    - the downside is that now we are reading from shared memory a lot to get shared parameters, and that we will need to communicate between threads to sum up the forces at the end
- solution 2: PxP data, Px1 threads
    - each thread computes for 1 destination from P sources
    - we get the benefit of sharing source params across threads, and don't have to share destination data or communicate between threads for reduction
        - **privatization**
        - fewer threads, more work per thread
        - communication between threads -> communication within a thread
        - less parallelisms (doesn't matter if we have an N large enough to keep the GPU busy).
    - Choosing P:
        - We want at least as many thread blocks as SMs, or else SMs will sit idle.
        - Bigger the P, the less memory bandwidth required (good)
        - Multiple blocks resident on one SM may be good
            - "potentially gives better latency hiding characteristics b/c you have more warps that may be in flight at the same time from slightly different pieces of the program"

Saw a trade off between: maximizing parallelism and more work/thread. This is b/c communicating within a thread is better than between threads

# SpMv

- what is the role of a thread?
- Imbalance in work if we have 1 thread per row (runtime is proportional to longest row )
    - but if there is no imbalance, there is a ~3x speed advantage over doing 1 thread per element since you don't have to communicate between threads
- Headscratcher: 1 thread per element or row when we don't know what the matrix looks like?
- Hybrid approach from 2009 [Supercomputing conference](http://sc15.supercomputing.org), Bell & Garland
    - solve the regular part of matrix with thread per row, and irregular with thread per element
    - is in [cuSparse library](http://docs.nvidia.com/cuda/cusparse/#axzz3iG51rxgn)

Big picture:

- Fine grained load imbalance means thread hardware sits idle
- registers > shared mem > global (for speed)

# Breadth first traversal

- DFS: less state
- BFS: more parallelism

- What we want in an algorithm
    - lots of parallelism
    - coalesced memory access
    - minimal execution divergence
    - easy to implement
- but if we have to do too much work (O(N^2), then its still a bad algorithm)

# Problem Set 6

[Seamless image cloning](http://www.ctralie.com/Teaching/PoissonImageEditing/):

- also called Poisson Image Editing
- given image A and B, and a mask for where to put B in A: blend the two images
- preserve the colors of image A (absolute information), and the gradient (relative information) of image B in A
    - gradient = differences between pixel and its neighbors
- Get a system of linear equations for H, our ideal composite image
    - can solve for H with Jacobi method (special case of gradient descent)
    - more in [the paper](http://www.cs.princeton.edu/courses/archive/fall10/cos526/papers/perez03.pdf)

Double buffering

- store approximate solution in buffer A, then compute better solution in B using A, then back again
- a popular parallel method
- find unknown pixel values inside the mask
- I_0 is our 1st guess
- Compute better guess: I_{k+1} = (A+B+C)/D
    A. sum of i_k's neighbors
    B. Sum of t's neighbors
    C. difference between s, and s's neighbors
    D. number of i_k's neighbors
