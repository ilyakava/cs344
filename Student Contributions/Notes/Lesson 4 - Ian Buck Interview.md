General Manager of Computer Software at Nvidia

GPUs for non-video graphics related (simulation and problem solving) - supercomputing applications

Started CUDA team
Worked on [SGI Octanes](https://en.wikipedia.org/wiki/SGI_Octane) in undergrad with OpenGL - simulation applications

## [BrookGPU Project](https://en.wikipedia.org/wiki/BrookGPU)
- program on top of existing graphics APIs without thinking about triangles/pixels
- important for GPU computation commercially
- Direct X 9 days - could express real programs (shading)
- academia - hack graphics APIs to do more general purpose things - but it was really hard to program (though in demand - finance firms hired game programmers)
- Study fundamental programming model of GPUs - and find a better more general one to show to the programmer
- Wrote: ray-tracer, triangle-meshing, grid simulation, flow, heat transfer

## CUDA project (2 years of works after Brook)

- latched onto basic idea of threading with programming model
- customers: didn't want to learn new language, easy access to performance
    - for that reason, no new parallel language, billed as extension of C

## Cool application:
- CT scanners - composite 2D scans to 3D model
- minimize x-ray radiation for view (20x so far)

## New ideas came into CUDA since launch

- support for managing concurrency, streams
- software support moved into hardware
- integrating closely with memory hierarchy
    - UVA: unify virtual address space (in CUDA 4), with operating system. System knows where memory is. Can dereference pointer without knowing if its on CPU or GPU, or another GPU

## What to do with 100x FLOPS

- some fields have limitless amounts of computational need
- Computational Disease: [Vijay Pande's Lab](https://pande.stanford.edu) at Stanford University

What would it mean if computation was free?

AstroGPU
- Conference: [2009](https://web.archive.org/web/20090729012622/http://www.astrogpu.org/)
- Blog; [2012](https://web.archive.org/web/20120624061002/http://www.astrogpu.org/)
