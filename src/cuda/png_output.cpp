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
      
      float* h_samples = (float *)calloc(bytes, 1); 
      
      cudaError_t error = cudaSuccess;
      error = cudaMemcpy(h_samples, memory_manager->d_samples, bytes, cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) {
         fprintf(stderr, "Failed to copy samples from device to host (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
      }
      
      // write them to file
      std::vector<unsigned char> data(4 * bytes);

      for (unsigned int i = 0; i < bytes; i++) {
         const unsigned int h_index = i;
         
         data[4 * i] = h_samples[h_index];
         data[4 * i + 1] = h_samples[h_index];
         data[4 * i + 2] = h_samples[h_index];
         data[4 * i + 3] = 255;
      }
//      
//      for (int y = 0; y < height; y++) {
//          for (int x = 0; x < width; x++) {
//
//            const unsigned int index = (y * width + x);
//
//            // TODO this needs 4 bytes per sample, we just have 1 right now
//            const auto r = static_cast<unsigned char>(h_samples[index]);
//            const auto g = static_cast<unsigned char>(h_samples[index]);
//            const auto b = static_cast<unsigned char>(h_samples[index]);
//            const auto a = static_cast<unsigned char>(255);
//
//            data[4 * index + 0] = r;
//            data[4 * index + 1] = g;
//            data[4 * index + 2] = b;
//            data[4 * index + 3] = a;
//         }
//      }


      unsigned lodepng_error = lodepng::encode("output.png", data, memory_manager->width, memory_manager->height);
      if (lodepng_error) {
         fprintf(stderr, "LodePNG encoding error (code %i): %s ", lodepng_error, lodepng_error_text(error));
         exit(1);
      }
   }
}
