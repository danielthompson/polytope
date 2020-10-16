//
// Created by daniel on 5/16/20.
//

#include <cuda_runtime.h>
#include "png_output.h"
#include "../../lib/lodepng.h"
#include "check_error.h"
#include <unistd.h>

namespace poly {
   void OutputPNG::Output(const poly::GPUMemoryManager* memory_manager) {
      // copy samples from device to host

      const unsigned int num_pixels = memory_manager->num_pixels;
      const unsigned int num_bytes = num_pixels * sizeof(float);
      
      float* h_samples_r = (float *)calloc(num_pixels, sizeof(float));
      float* h_samples_g = (float *)calloc(num_pixels, sizeof(float));
      float* h_samples_b = (float *)calloc(num_pixels, sizeof(float));
      
      cuda_check_error( cudaMemcpy(h_samples_r, memory_manager->host_samples.r, num_bytes, cudaMemcpyDeviceToHost) );
      cuda_check_error( cudaMemcpy(h_samples_g, memory_manager->host_samples.g, num_bytes, cudaMemcpyDeviceToHost) );
      cuda_check_error( cudaMemcpy(h_samples_b, memory_manager->host_samples.b, num_bytes, cudaMemcpyDeviceToHost) );
      
      // write them to file
      std::vector<unsigned char> data(4 * num_bytes);

      for (unsigned int i = 0; i < num_pixels; i++) {
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
      
      char* cwd = get_current_dir_name();
      
      printf("Output successful to %s\n", cwd);
      
      free(cwd);
   }
}
