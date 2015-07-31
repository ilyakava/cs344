cs344
=====

Introduction to Parallel Programming class code

# Building on OS X

These instructions are for OS X 10.9 "Mavericks", and have been tested on Ubuntu 14.10.

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

- [Udacity forum](https://discussions.udacity.com/c/standalone-courses/intro-to-parallel-programming)
- [upenn class](http://cis565-fall-2014.github.io)