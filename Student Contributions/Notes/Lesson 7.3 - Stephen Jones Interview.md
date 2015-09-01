Stephen jones

Cuda lead
programming model

Background not computing, never had a computing class. Fluid mechanics and classical physics. Graduated in 96, world was already moving into computers & science together. Wrote videogames, went to military for HPC, worked in startups, eventually came to Nvidia.

Dynamic Parallelism

Which problem trying to solve - make GPU easier to program and broaden what you can solve. Bulk parallelism was easy to express. Easier way to extract more parallelism in your problem.

Launch kernels from other kernels - analogue of spawn threads from a process on CPU. Dynamically/on the fly launch threads according to a specific logic situation, easier to bring parallelism to complex control structures or complex data structures.

No longer have to launch a piece of work from a state change on CPU, less managing to do. Also amortizes performance overheads. Makes it easier to overlap work on the GPU and keep it busy.

Any problem where amount of work to do is discovered in the processes of working. Be able to balance problems with irregular parallelism.

N-body simulation with dynamic parallelism (Oct-tree [cubes] is key component, no longer all-to-all O(n^2) problem, but O(nlogn) or O(n)). Tree build is 1/2 the time (this is where dynamic parallelism pays off. Not building tree level by level and waste time on sparse regions, but focus instead on dense regions).

Next generation of CUDA. Heterogeneity problem (you've got 2 subproblems, 1 parallel and 1 serial, how do you write code for that problem with the greatest ease?). Memory models (not always knowing what memory to move, or when its inconvenient to move data between computations).

Biggest customer problems, (not MPI is hard, Clusters are hard, Networking is hard), but parallel programming is hard. Advice to students, get used to thinking in parallel, b/c that is the hardest part.
