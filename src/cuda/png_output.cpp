//
// Created by daniel on 5/16/20.
//

#include <cuda_runtime.h>
#include "png_output.h"
#include "../../lib/lodepng.h"
#include "check_error.h"
#include <unistd.h>

namespace poly {
   void output_png::output(const poly::render_context *render_context, const std::string& filename) {
      // copy samples from device to host

      const unsigned int num_pixels = render_context->total_pixel_count;
      const unsigned int num_bytes = num_pixels * sizeof(float);
      
      float* h_samples_r = (float *)calloc(num_pixels, sizeof(float));
      float* h_samples_g = (float *)calloc(num_pixels, sizeof(float));
      float* h_samples_b = (float *)calloc(num_pixels, sizeof(float));
      
      for (int device_index = 0; device_index < render_context->device_count; device_index++) {
         const poly::device_context* device_context = &render_context->device_contexts[device_index];
         
         const size_t channel_byte_count = device_context->pixel_count * sizeof(float);
         
         float* temp_r = (float *)malloc(channel_byte_count);
         float* temp_g = (float *)malloc(channel_byte_count);
         float* temp_b = (float *)malloc(channel_byte_count);
         
         cuda_check_error(cudaSetDevice(device_index));
         cuda_check_error( cudaMemcpy(temp_r, device_context->host_samples.r, channel_byte_count, cudaMemcpyDeviceToHost) );
         cuda_check_error( cudaMemcpy(temp_g, device_context->host_samples.g, channel_byte_count, cudaMemcpyDeviceToHost) );
         cuda_check_error( cudaMemcpy(temp_b, device_context->host_samples.b, channel_byte_count, cudaMemcpyDeviceToHost) );
         
         const size_t tile_row_byte_count = device_context->width * sizeof(float);
         for (int i = 0; i < device_context->height; i++) {
            const size_t dest_offset = render_context->width * i + device_context->width * device_index;
            const size_t src_offset = device_context->width * i;
            memcpy(h_samples_r + dest_offset, temp_r + src_offset, tile_row_byte_count);
            memcpy(h_samples_g + dest_offset, temp_g + src_offset, tile_row_byte_count);
            memcpy(h_samples_b + dest_offset, temp_b + src_offset, tile_row_byte_count);
         }
      }
      
      // write them to file
      std::vector<unsigned char> data(4 * num_bytes);

      for (unsigned int i = 0; i < num_pixels; i++) { 
         const unsigned int h_index = i;
         
         data[4 * i] = h_samples_r[h_index] > 255 ? 255 : h_samples_r[h_index];
         data[4 * i + 1] = h_samples_g[h_index] > 255 ? 255 : h_samples_g[h_index];
         data[4 * i + 2] = h_samples_b[h_index] > 255 ? 255 : h_samples_b[h_index];
         data[4 * i + 3] = 255;
      }

      Log.debug("Writing output file...");
      unsigned lodepng_error = lodepng::encode(filename.c_str(), data, render_context->width, render_context->height);
      if (lodepng_error) {
         ERROR("LodePNG encoding error (code %i): %s ", lodepng_error, lodepng_error_text(lodepng_error));
      }
      
      char* cwd = get_current_dir_name();
      
      Log.info("output successful to %s/%s", cwd, filename.c_str());
      
      free(cwd);
   }
}
