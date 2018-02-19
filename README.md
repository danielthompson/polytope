# Polytope
Hobby ray/path tracer in C++. 

#### Goals
* Ray tracer
* Path tracer
* OpenGL visualization
* Realtime CUDA view
* Acceleration structures
  * BVH
  * kd-tree


#### What's a polytope?

A cool sciency word I found on Wikipedia.

> _In elementary geometry, a polytope is a geometric object with "flat" sides. It is a generalisation in any number of dimensions of the three-dimensional polyhedron. Polytopes may exist in any general number of dimensions n as an n-dimensional polytope or n-polytope._ 
> 
> https://en.wikipedia.org/wiki/Polytope

#### Does polytope render polytopes?

Sort of. It will render a specific subset (triangles) of 2-polytopes (polygons), and a specific subset (cuboids, triangle meshes) of 3-polytopes (polyhedra). 

It will also render a bunch of non-polytopes, like spheres, disks, cylinders, tori, quadrics, etc.