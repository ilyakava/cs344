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
    - tree methods ([Barnes-Hut](https://en.wikipedia.org/wiki/Barnes–Hut_simulation))
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

Saw a trade off between: maximizing parallelism and **more work/thread**. This is because communicating within a thread is better than between threads (**minimize global mem bandwidth**).

# SpMv

- right data structure can make a big difference
- what is the role of a thread?
- Imbalance in work if we have 1 thread per row (runtime is proportional to longest row )
    - but if there is **no imbalance**, there is a ~3x speed advantage over doing 1 thread per element since you don't have to communicate between threads
- Headscratcher: 1 thread per element or row when we don't know what the matrix looks like?
- Hybrid approach from 2009 [Supercomputing conference](http://sc15.supercomputing.org), Bell & Garland
    - solve the regular part of matrix with thread per row, and irregular with thread per element
    - is in [cuSparse library](http://docs.nvidia.com/cuda/cusparse/#axzz3iG51rxgn)

Big picture:

- Fine grained load imbalance means thread hardware sits idle
- registers > shared mem > global (for speed)

# Breadth first traversal (6.1 ends and 6.2 continues)

- DFS: less state
- BFS: more parallelism

- What we want in an algorithm
    - lots of parallelism
    - coalesced memory access
    - minimal execution divergence
    - easy to implement
- but if we have to do too much work (O(N^2), then its still a bad algorithm)
    - serial is efficient b/c there is a frontier that keeps work to O(N)
- how do we do it in parallel. **Choosing a better algorithm**
    - store (directed) graph with CSR type structure ([compressed sparse row](http://www.cs.washington.edu/research/projects/uns/FC5/src/boost_1_40_0/libs/graph/doc/compressed_sparse_row.html))
        - C array: in cardinal node order, for each edge connected to a node, list the id of the node the edge points to
        - R array: the vth entry points to the start of the vth node's connections in the C array
        - D array: the vth entry is the depth of the vth node
    - Merrill's Algorithm:
        1. in parallel, for each node on the frontier, find starting point of its neighbors: `R[v]`
        2. for each frontier node, calculate its number of neighbors: `R[v+1] - R[v]`
        3. Allocate space to store the new frontier (scan)
        4. copy each active edge list (potential new frontier)
        5. Cull the visited vertexes (compact)
    - Linear work, ends up 4x faster than CPU operation
- List ranking (good ex of a not inherently parallel problem)
    - worst possible graph for parallelism is a linked list
    - we want to traverse from every node (n iterations on each of n nodes)
    - ideal strategy to taking advantage of GPUs: same work complexity, but smaller step complexity
        - **compromise with more work for fewer steps** (since unless we overload our many parallel processors, our runtime grows with step size)
        - using chum pointer -> chum gets us nlogn (reuse past work)
            - Hillis & Steele
            - general way to get a speedup (sorting a circular list example)
- Hash table
    - CPU: within a bucket, chain linked list of values that you have to traverse
        - in a parallel setting, chaining is bad:
            1. load imbalance
                - slowest thread in the warp is the limiting factor
            2. contention in construction
                - simultaneous bucket updates
    - Cuckoo hashing (skip the **serial data structure**)
        - bad bird (lays eggs in another bird's nest)
        - have multiple hash tables/functions, iterate to kick out old items (must be atomic) in tables until all items are hashed, give up after some limit of iterations if its not possible
            - lookup in multiple tables (have to know when of the table lookups is right) although they are constant time (keeps all threads busy)

# Problem Set 6 - [Seamless image cloning](http://www.ctralie.com/Teaching/PoissonImageEditing/):

- also called Poisson Image Editing
- given image A and B, and a mask for where to put B in A: blend the two images
- preserve the colors of image A (absolute information), and the gradient (relative information) of image B in A
    - gradient = differences between pixel and its neighbors
- Get a system of linear equations for H, our ideal composite image
    - can solve for H with Jacobi method (special case of gradient descent)
    - more in [the paper](http://www.cs.princeton.edu/courses/archive/fall10/cos526/papers/perez03.pdf)
- I_0 is our 1st guess
- Compute better guess: I_{k+1} = (A+B+C)/D
    A. sum of i_k's neighbors
    B. Sum of t's neighbors
    C. difference between s, and s's neighbors
    D. number of i_k's neighbors

Double buffering

- store approximate solution in buffer A, then compute better solution in B using A, then back again
- a popular parallel method
- find unknown pixel values inside the mask

