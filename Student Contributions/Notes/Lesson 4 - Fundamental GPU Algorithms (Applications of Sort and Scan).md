Easy way to parallelize recurrence relationships.
Scan is efficient on GPUs thanks to Blelloch, and so its the core of a number of interesting parallel primitives

# Scan

- running total, max min
- compact (i.e. filter)
    - subsets filtered by predicate
        - input is long array
        - predicate (object input, T/F output)
        - scatter address (place in output array) is intermediate
        - output is an array with fewer elements
        - sparse (index preserving with null) or dense output possible
            - in general dense is better (might save threads in a lock environment)
        - input is scattered to some addresses
            - example of rendering/clipping triangles
            - this is exclusive scan (generalizing compact)
            - input is array of each shape and how many triangles it wants to create
            - output is address/location in array of where to write the triangles
- segmented scan
    - want lots of work in a single launched kernel, so we'll pack a few small scans together, and then have an independent scan operator
    - scan, but starting over at each segment
    - has same complexity as your choice of unsegmented scan algorithm (but more memory traffic)
- Sparse matrix dense vector (SpMv)
    - sparse (doesn't store 0's)
    - compressed sparse row (CSR) representation
        - value (1d array of values)
            - matrix compressed into flat segmented array
        - column (array of column indexes)
            - which vector element to multiply by (gather vector values into flat array corresponding to flat matrix array)
        - rowptr (which element does each row begin with)
            - where segments in flat array exist

# Sort

- many for serial, few for parallel
    - branchy, moving around little memory at a time
- we need:
    - coalescing
    - few branch divergence
    - keep lots of threads busy simultaneously

- some algorithms map nicely to parallel world

## odd-even/brick (parallel version of bubble)

## merge sort

- hard part: number and size of merges differs greatly
- 3 stages
    1. n block sort. tons of tasks (often not even use merge sort here, many blocks we need to sort)
        - many more problems than SMs
    2. merge lots of small sorted blocks, 1 merge per 1 thread block
        - like compact
            - have 2 lists of length n
            - launch 2n threads
                - calculate position of element in final list
                    - sum of positions in current list, and 2nd list
                        - will need binary search (logn) in other list, fast in shared memory
                - and scatter it
    3. few, or one, big task (long lists to merge)
        - bad b/c lots of SMs will be idle since we only have 1 task to do
        - so we want to split the work into smaller merges, so that no task is too big for 1 SM
        - select each **nth** element, splitters, merge them, and sort
            - find out the position of a splitter, in the other list
        - what we get is that the list of elements between any two splitters can be independently sent to an SM
            - length of this is at most **2n**

## sorting networks

- oblivious sorting algorithms (no data dependence except for a swap operation)
    - CPU advantage is complex control flow, while GPU advantage is lots of simple control flow, so oblivious is a good match for GPU
- bitonic sequence - only changes direction once
    - easy to sort (b/c its easy to split 1 bitonic sequence into 2 bitonic sequences)
    - logn stages required
    - nlogn work
- bitonic network, scales well
- for sorting networks, no matter the input, same time
- odd even sort is even more efficient than bitonic sorting


## GPU performance leader: radix sort [pronounced: raid-ichs]

- [algorithm](http://stackoverflow.com/a/31840945/2256243):
    - start with least significant bit (LSB)
    - split into two sets based on it (otherwise preserving order)
        - get the predicate by anding with a number that has that bit as 1, and every other bit as 0
        - scan that to get the addresses to scatter the lower numbers
        - scan again to get the addresses to scatter the higher numbers
        - common optimization: take 4 bits per pass, and get a 16 way split
    - move to NSB and repeat
- best comparison based sorts are nlogn, but this one is O(kn) (linear in bits of representation, and number of items to sort). Work complexity is usually proportional to number of items to sort
- brute force, but simple and fast

## Quicksort

- more complex on GPUS
- generally:
    - choose single pivot
    - split into 3 arrays (<, =, >)
    - recurse
        - makes it complicated, GPUs only recently started to support recursion (we don't have it in this class)
            - recursion was added in compute capability 2.0 and CUDA 3.1
        - without recursion: segments

## key value sorts

## Problem Set 4 - Radix Sort

Factors nicely into 3 different parallel operations:
1. stencil
2. **sort**
3. map

- compute score, a likelihood of a pixel belonging to a red-eye
    - normalized cross-correlation (stencil)
- sort these
- reduce redness
    - map
