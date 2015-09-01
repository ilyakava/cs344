B.13. Warp Vote Functions

A single method to do something in every warp: `__ballot`

---

64 KB of constant memory cache per SM, works best if all threads in a warp access same location. Variables exist for lifespan of the application

---

48 KB of read-only cache per Kepler SM is also available (constant memory). Writable by host, readable by device. Separate memory bandwidth from global memory makes this faster. Good for scattered reads.