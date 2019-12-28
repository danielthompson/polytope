# Polytope  
Hobby ray/path tracer in C++.  

#### Build Status

[![Build Status](https://travis-ci.org/danielthompson/polytope.svg?branch=master)](https://travis-ci.org/danielthompson/polytope) [![Coverage Status](https://coveralls.io/repos/github/danielthompson/polytope/badge.svg?branch=master)](https://coveralls.io/github/danielthompson/polytope?branch=master)

#### Goals
* Path tracer
* Support a variety of materials and reflection / transmission models
* Acceleration structures
  * BVH
  * kd-tree
* Import a variety of scene / object description grammars
  * pbrt
  * obj
  * ply
* Rasterization output path with OpenGL (or Vulkan)
* Stream processing, perhaps with CUDA and/or ISPC

#### What's a polytope?

> _In elementary geometry, a polytope is a geometric object with "flat" sides. It is a generalisation in any number of dimensions of the three-dimensional polyhedron. Polytopes may exist in any general number of dimensions n as an n-dimensional polytope or n-polytope._ 
> 
> https://en.wikipedia.org/wiki/Polytope

#### Does polytope render polytopes?

Polytope represents all shapes internally as triangle meshes, which are 3-polytopes. Input grammars that specify other types of geometry (implicit surfaces, etc) are tesselated on import to triangle meshes.

So, yes.

#### Attribution

The structure and terminology of this project is inspired by pbrt (https://github.com/mmp/pbrt-v3). My MO is generally to:
 1. Originally develop and implement a feature to the point that it works; and then
 2. See how the pros did it; and then
 3. Adjust my own implementation if necessary / desired.

This project uses code from the following libraries / projects:

 * pbrt
   * https://github.com/mmp/pbrt-v3
   * Matrices and transforms
 * Google Test
   * https://github.com/google/googletest
   * Unit testing
 * LodePNG
   * https://github.com/lvandeve/lodepng
   * For exporting raw image data to PNG
   