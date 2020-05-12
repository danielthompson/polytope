#include <cstdio>
#include "../../structures/Ray.h"

float* d_x;
float* d_y;
float* d_z;
int num_vertices;

__global__ void linear_intersect_kernel(Polytope::Ray &ray) {
   
}

void linear_intersect(Polytope::Ray &ray) {
   const int threadsPerBlock = 256;
   const int blocksPerGrid = (num_vertices + threadsPerBlock - 1) / threadsPerBlock;
   linear_intersect_kernel<<<blocksPerGrid, threadsPerBlock>>>(ray);
}

void initialize_unpacked_mesh(const float *h_x, const float *h_y, const float *h_z, const int num_verts) {
   num_vertices = num_verts;
   
   cudaError_t err;
   size_t size = num_verts * sizeof(float);
   
   err = cudaMalloc((void **)&d_x, size);

   if (err != cudaSuccess)
   {
      fprintf(stderr, "Failed to allocate device vector x (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }

   err = cudaMalloc((void **)&d_y, size);

   if (err != cudaSuccess)
   {
      fprintf(stderr, "Failed to allocate device vector y (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }

   err = cudaMalloc((void **)&d_z, size);

   if (err != cudaSuccess)
   {
      fprintf(stderr, "Failed to allocate device vector z (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }

   err = cudaMemcpy(d_x, h_x, size, ::cudaMemcpyHostToDevice);

   if (err != cudaSuccess)
   {
      fprintf(stderr, "Failed to copy vector x from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }

   err = cudaMemcpy(d_y, h_y, size, ::cudaMemcpyHostToDevice);

   if (err != cudaSuccess)
   {
      fprintf(stderr, "Failed to copy vector y from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }

   err = cudaMemcpy(d_z, h_z, size, ::cudaMemcpyHostToDevice);

   if (err != cudaSuccess)
   {
      fprintf(stderr, "Failed to copy vector z from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}



void free_mesh() {
   cudaError_t err;
   err = cudaFree(d_x);
   err = cudaFree(d_y);
   err = cudaFree(d_z);
}