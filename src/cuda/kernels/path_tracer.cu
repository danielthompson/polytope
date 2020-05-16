//
// Created by daniel on 5/15/20.
//

#include "path_tracer.cuh"

namespace Polytope {
   
   __global__ void path_trace_kernel(const unsigned int width, const unsigned int height, float* d_samples) {
      // loop over pixels
      const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
      const unsigned int pixel_x = index % width;
      const unsigned int pixel_y = index / width;
      
      // temp
      d_samples[index] = 127.f;
   }
   
   void PathTracerKernel::Trace() {

      constexpr unsigned int width = 640;
      constexpr unsigned int height = 480;
      constexpr unsigned int num_pixels = width * height;
      
      constexpr unsigned int threadsPerBlock = 256;
      constexpr unsigned int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

      cudaError_t error = cudaSuccess;
      path_trace_kernel<<<blocksPerGrid, threadsPerBlock>>>(width, height, memory_manager->d_samples);

      cudaDeviceSynchronize();
      error = cudaGetLastError();

      if (error != cudaSuccess)
      {
         fprintf(stderr, "Failed to launch generate_ray_kernel (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      //Intersection intersection = Scene->GetNearestShape(current_ray, x, y);
   }
}
