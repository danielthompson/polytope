//
// Created by daniel on 5/13/20.
//

#include <iostream>
#include <chrono>
#include <sstream>
#include "../common/utilities/Common.h"
#include "../common/utilities/OptionsParser.h"
#include "../common/structures/point2.h"
#include "../common/parsers/pbrt_parser.h"

#include "context.h"
#include "kernels/path_tracer.cuh"
#include "png_output.h"
#include "../common/utilities/thread_pool.h"
#include "check_error.h"

poly::Logger Log;
struct poly::stats main_stats;
thread_local struct poly::stats thread_stats;

float roundOff(float n) {
   float d = n * 100.0f;
   int i = d + 0.5;
   d = (float)i / 100.0f;
   return d;
}

std::string convertToString(float num) {
   std::ostringstream convert;
   convert << num;
   return convert.str();
}

std::string convertSize(size_t size) {
   static const char *SIZES[] = { "B/s", "KB/s", "MB/s", "GB/s" };
   int div = 0;
   size_t rem = 0;

   while (size >= 1024 && div < (sizeof SIZES / sizeof *SIZES)) {
      rem = (size % 1024);
      div++;
      size /= 1024;
   }

   float size_d = (float)size + (float)rem / 1024.0f;
   std::string result = convertToString(roundOff(size_d)) + " " + SIZES[div];
   return result;
}



int main(int argc, char* argv[]) {
   try {
      Log = poly::Logger();
      LOG_INFO("Polytope (CUDA) started.");

      int num_cuda_devices = 0;
      cuda_check_error(cudaGetDeviceCount(&num_cuda_devices));
      
      if (num_cuda_devices < 1) {
         LOG_ERROR("No CUDA devices detected. Exiting.");
         exit(1);
      }
      
      poly::render_context render_context { };

      LOG_INFO("Detected " + std::to_string(num_cuda_devices) + " CUDA devices:");
      for (int i = 0; i < num_cuda_devices; i++) {
         cudaDeviceProp cuda_device_prop{};
         cuda_check_error(cudaGetDeviceProperties(&cuda_device_prop, i));
         LOG_INFO("(" << i << ") " << cuda_device_prop.name << " - " << cuda_device_prop.totalGlobalMem / (1000000000ul));
      }
      
      render_context.device_count = num_cuda_devices;
      
      poly::Options options = poly::Options();

      if (argc > 0) {
         poly::OptionsParser parser(argc, argv);
         options = parser.Parse();
      }

      if (options.help) {
         std::cout << "Polytope (CUDA version) by Daniel A. Thompson, built on " << __DATE__ << std::endl;
         fprintf(stderr, R"(
Usage: polytope_cuda [options] -inputfile <filename> [-outputfile <filename>]

Rendering options:
   -samples <n>      Number of samples to use per pixel. Optional; overrides
                     the number of samples specified in the scene file.

File options:
   -inputfile        The scene file to render. Currently, PBRT is the only
                     supported file format. Required.
   -outputfile       The filename to render to. Currently, PNG is the only
                     supported output file format. Optional; overrides the
                     output filename specified in the scene file, if any;
                     defaults to the input file name (with .png extension).

Other:
   --help            Print this help text and exit.)");
         std::cout << std::endl;
         exit(0);
      }

      const auto totalRunTimeStart = std::chrono::system_clock::now();

//      LOG_INFO("Using 1 CPU thread.");

      {
         if (!options.inputSpecified) {
            ERROR("No input file specified.");
         }
         
         // load file
         poly::pbrt_parser parser = poly::pbrt_parser();
         const auto runner = parser.parse_file(options.input_filename);
         
         // override parsed with options here
         if (options.samplesSpecified) {
            runner->sample_count = options.samples;
         }

         LOG_DEBUG("Image is [" << runner->Bounds.x << "] x [" << runner->Bounds.y << "], " << runner->sample_count << " spp.");

         const auto bound_start = std::chrono::system_clock::now();
         thread_stats.num_bvh_bound_leaf_same_centroid = 0;
         unsigned int num_nodes = runner->Scene->bvh_root.bound(runner->Scene->Shapes);
         const auto bound_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> bound_duration = bound_end - bound_start;
         LOG_DEBUG("Created BVH with " << add_commas(num_nodes) << " nodes in " << std::to_string(bound_duration.count()) << "s.");

         const auto compact_start = std::chrono::system_clock::now();
         runner->Scene->bvh_root.compact();
         const auto compact_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> compact_duration = compact_end - compact_start;
         LOG_DEBUG("Compacted BVH in " << compact_duration.count() << "s.");

         render_context.width = runner->Scene->Camera->settings.bounds.x;
         render_context.height = runner->Scene->Camera->settings.bounds.y;
         render_context.total_pixel_count = render_context.width * render_context.height;
         
         poly::thread_pool thread_pool(render_context.device_count);
         
         for (int device_index = 0; device_index < render_context.device_count; device_index++) {
            thread_pool.enqueue([&, device_index] {
               cuda_check_error(cudaSetDevice(device_index));
               render_context.device_contexts.emplace_back(render_context.width / 2, render_context.height, device_index);
               LOG_DEBUG("[%i] - Copying data to GPU", device_index);
               const auto copy_start_time = std::chrono::system_clock::now();
               size_t bytes_copied = render_context.device_contexts[device_index].malloc_scene(runner->Scene);
               const auto copy_end_time = std::chrono::system_clock::now();
               const std::chrono::duration<double> copy_duration = copy_end_time - copy_start_time;

               float effective_bandwidth = ((float) bytes_copied) / (float) copy_duration.count();
               std::string bandwidth_string = convertSize((size_t) effective_bandwidth);
               std::string comma_string = add_commas(bytes_copied);
               LOG_DEBUG("[" << device_index << "] - Copied " << add_commas(bytes_copied) << " bytes in " <<
                         copy_duration.count() << " (" << bandwidth_string << ").");

               LOG_INFO("[" << device_index << "] - Rendering with " << runner->sample_count << "spp...");

               poly::path_tracer path_tracer_kernel(&render_context.device_contexts[device_index]);
               const auto render_start_time = std::chrono::system_clock::now();
               path_tracer_kernel.Trace(runner->sample_count);
               const auto render_end_time = std::chrono::system_clock::now();
               const std::chrono::duration<double> render_duration = render_end_time - render_start_time;
               LOG_INFO("[" << device_index << "] - Sampling complete in " << render_duration.count() << "s.");
            });
         }
         
         thread_pool.synchronize();
         
         
         LOG_INFO("Outputting samples to film...");
         const auto output_start_time = std::chrono::system_clock::now();
         poly::output_png::output(&render_context, options.output_filename);
         const auto output_end_time = std::chrono::system_clock::now();
         const std::chrono::duration<double> output_duration = output_end_time - output_start_time;
         LOG_INFO("Output complete in " << output_duration.count() << "s.");
      }

      const auto totalRunTimeEnd = std::chrono::system_clock::now();
      const std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;

      LOG_INFO("Total computation time: " << totalElapsedSeconds.count() << "s.");
      LOG_INFO("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
