//
// Created by daniel on 5/16/20.
//

#include <cuda_runtime.h>
#include "png_output.h"
#include "../../lib/lodepng.h"

namespace Polytope {
   void OutputPNG::Output(const std::shared_ptr<Polytope::GPUMemoryManager>& memory_manager) {
      // copy samples from device to host

      const unsigned int bytes = sizeof(float) * memory_manager->num_pixels;
      
      float* h_samples_r = (float *)calloc(bytes, 1);
      float* h_samples_g = (float *)calloc(bytes, 1);
      float* h_samples_b = (float *)calloc(bytes, 1);
      
      cudaError_t error = cudaSuccess;
      error = cudaMemcpy(h_samples_r, memory_manager->d_samples_r, bytes, cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
         fprintf(stderr, "Failed to copy samples_r from device to host (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }

      error = cudaSuccess;
      error = cudaMemcpy(h_samples_g, memory_manager->d_samples_g, bytes, cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
         fprintf(stderr, "Failed to copy samples_g from device to host (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }

      error = cudaSuccess;
      error = cudaMemcpy(h_samples_b, memory_manager->d_samples_b, bytes, cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
         fprintf(stderr, "Failed to copy samples_b from device to host (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      // write them to file
      std::vector<unsigned char> data(4 * bytes);

      for (unsigned int i = 0; i < bytes; i++) {
         const unsigned int h_index = i;
         
         data[4 * i] = h_samples_r[h_index] > 255 ? 255 : h_samples_r[h_index];
         data[4 * i + 1] = h_samples_g[h_index] > 255 ? 255 : h_samples_g[h_index];
         data[4 * i + 2] = h_samples_b[h_index] > 255 ? 255 : h_samples_b[h_index];
         data[4 * i + 3] = 255;
      }

      unsigned lodepng_error = lodepng::encode("output.png", data, memory_manager->width, memory_manager->height);
      if (lodepng_error) {
         fprintf(stderr, "LodePNG encoding error (code %i): %s ", lodepng_error, lodepng_error_text(error));
         exit(1);
      }
   }
}
