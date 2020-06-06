//
// Created by daniel on 5/16/20.
//

#include <cuda_runtime.h>
#include "png_output.h"
#include "../../lib/lodepng.h"
#include "check_error.h"

namespace Polytope {
   void OutputPNG::Output(const Polytope::GPUMemoryManager* memory_manager) {
      // copy samples from device to host

      const unsigned int bytes = sizeof(float) * memory_manager->num_pixels;
      
      float* h_samples_r = (float *)calloc(bytes, 1);
      float* h_samples_g = (float *)calloc(bytes, 1);
      float* h_samples_b = (float *)calloc(bytes, 1);
      
      
      cuda_check_error( cudaMemcpy(h_samples_r, memory_manager->host_samples.r, bytes, cudaMemcpyDeviceToHost) );
      cuda_check_error( cudaMemcpy(h_samples_g, memory_manager->host_samples.g, bytes, cudaMemcpyDeviceToHost) );
      cuda_check_error( cudaMemcpy(h_samples_b, memory_manager->host_samples.b, bytes, cudaMemcpyDeviceToHost) );
      
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
         fprintf(stderr, "LodePNG encoding error (code %i): %s ", lodepng_error, lodepng_error_text(lodepng_error));
         exit(1);
      }
   }
}
