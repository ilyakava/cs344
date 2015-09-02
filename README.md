[cs344](https://www.udacity.com/course/intro-to-parallel-programming--cs344)
=====

Introduction to Parallel Programming Udacity class code

Includes student notes for [entire course](https://github.com/ilyakava/cs344/tree/master/Student%20Contributions/Notes), solutions to all [Problem sets](https://github.com/ilyakava/cs344/tree/master/Problem%20Sets), completed code segments of the [Final Assessment](https://github.com/ilyakava/cs344/tree/master/Final).

# Building on OS X

These instructions are for OS X 10.9 "Mavericks", and have also been successfully tested on Ubuntu 14.10.

* Step 1. Build and install OpenCV. The best way to do this is with
Homebrew. However, you must slightly alter the Homebrew OpenCV
installation; you must build it with libstdc++ (instead of the default
libc++) so that it will properly link against the nVidia CUDA dev kit.
[This entry in the Udacity discussion forums](http://forums.udacity.com/questions/100132476/cuda-55-opencv-247-os-x-maverick-it-doesnt-work) describes exactly how to build a compatible OpenCV.
  * **NOTE:** as of cuda 7.0 this is no longer necessary. See [my script](https://gist.github.com/ilyakava/6f22d458b9771e7ccc97) for opencv installation details.

* Step 2. You can now create 10.9-compatible makefiles, which will allow you to
build and run your homework on your own machine:
```
mkdir build
cd build
cmake ..
make
```

# Links

Key:
- :closed_book: = link to a book
- :book: = link to online/free book

- [Udacity forum](https://discussions.udacity.com/c/standalone-courses/intro-to-parallel-programming)
- Other courses
    - [Saraviensis ws1213](https://graphics.cg.uni-saarland.de/fileadmin/cguds/courses/ws1213/pp_cuda/)
    - [UC Davis EEC 171](http://www.nvidia.com/object/cudau_ucdavis)
        - taught by John Owens
        - [:closed_book: Computer Architecture: A Quantitative Approach, 4th Ed](http://www.amazon.com/Computer-Architecture-Quantitative-Approach-Edition/dp/0123704901)
    - [UC Davis EEC 277](https://smartsite.ucdavis.edu/portal/site/41cdf6c8-0223-40c9-a69a-1543d7ea2575/page/56af4247-af49-4a77-8c79-df649d564d7c)
        - taught by John Owens
        - [:closed_book: Real time rendering](http://www.amazon.com/Real-Time-Rendering-Third-Tomas-Akenine-Moller/dp/1568814240/ref=sr_1_1?s=books&ie=UTF8&qid=1438705376&sr=1-1&keywords=real+time+rendering)
        - [:closed_book: Texture and Modeling, a procedural approach](http://www.amazon.com/Texturing-Modeling-Third-Procedural-Approach/dp/1558608486/ref=sr_1_1?s=books&ie=UTF8&qid=1438705410&sr=1-1&keywords=texturing+and+modeling+a+procedural+approach)
        - [:closed_book: Cg tutorial](http://www.amazon.com/The-Tutorial-Definitive-Programmable-Real-Time/dp/0321194969)
    - [UPenn CIS 565](http://cis565-fall-2014.github.io)
        - [:closed_book: Programming Massively Parallel Processors](http://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0123814723)
        - [:closed_book: Real time rendering](http://www.amazon.com/Real-Time-Rendering-Third-Tomas-Akenine-Moller/dp/1568814240/ref=sr_1_1?s=books&ie=UTF8&qid=1438705376&sr=1-1&keywords=real+time+rendering)
    - [UC Davis ECS 223](http://web.cs.ucdavis.edu/~amenta/s13/parallel.html)
        - guest lectures by John Owens
        - [:closed_book: Introduction to Parallel Algorithms](http://www.amazon.com/Introduction-Parallel-Algorithms-Joseph-JaJa/dp/0201548569)
    - [Coursera heterogeneous parallel programming](https://www.coursera.org/course/hetero)
        - [:closed_book: Programming Massively Parallel Processors](http://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0123814723)
- Software
    - [ML links with SVMs](http://fastml.com/running-things-on-a-gpu/)
    - [kernel machines](http://mloss.org/software/view/236/)
        - [@ UMD](http://www.umiacs.umd.edu/~ramani/)
        - [other mloss](http://mloss.org/software/search/?searchterm=GPU&post=)
- Additional Books
    - GPU gems:
        - [:book: 1](http://http.developer.nvidia.com/GPUGems/gpugems_part01.html)
        - [:book: 2](http://http.developer.nvidia.com/GPUGems2/gpugems2_frontmatter.html)
        - [:book: 3](http://http.developer.nvidia.com/GPUGems3/gpugems3_pref01.html)
    - NVLabs [:book: Design patterns for GPU computing](http://nvlabs.github.io/moderngpu/)
    - [:book: Introduction to Parallel Computing (Vipin Kumar)](http://parallelcomp.uw.hu/index.html)
- Conferences
    - [Nvidia GPU Tech Conf](http://www.gputechconf.com/attend/sessions)
