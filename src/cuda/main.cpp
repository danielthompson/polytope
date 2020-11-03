//
// Created by daniel on 5/13/20.
//

#include <iostream>
#include <chrono>
#include <sstream>
#include <atomic>
#include "../common/utilities/Common.h"
#include "../common/utilities/OptionsParser.h"
#include "../common/structures/Point2.h"
#include "../common/parsers/pbrt_parser.h"

#include "gpu_memory_manager.h"
#include "kernels/path_tracer.cuh"
#include "png_output.h"
#include "mesh/cuda_mesh_soa.h"
#include "../cpu/structures/stats.h"

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
      Log.info("Polytope (CUDA) started.");

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

      Log.info("Using 1 CPU thread.");

      {
         if (!options.inputSpecified) {
            ERROR("No input file specified.");
         }
         
         // load file
         poly::pbrt_parser parser = poly::pbrt_parser();
         const auto runner = parser.parse_file(options.input_filename);
         
         
         
         // override parsed with options here
         if (options.samplesSpecified) {
            runner->NumSamples = options.samples;
         }
         
         Log.info("Image is [%i] x [%i], %i spp.", runner->Bounds.x, runner->Bounds.y, runner->NumSamples);
         
         const auto bound_start = std::chrono::system_clock::now();
         thread_stats.num_bvh_bound_leaf_same_centroid = 0;
         unsigned int num_nodes = runner->Scene->bvh_root.bound(runner->Scene->Shapes);
         const auto bound_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> bound_duration = bound_end - bound_start;
         Log.debug("Created BVH with " + add_commas(num_nodes) + " nodes in " + std::to_string(bound_duration.count()) + "s.");

         const auto compact_start = std::chrono::system_clock::now();
         runner->Scene->bvh_root.compact();
         const auto compact_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> compact_duration = compact_end - compact_start;
         Log.debug("Compacted BVH in %f s.", compact_duration.count());
         Log.debug("Copying data to GPU...");
         const auto copy_start_time = std::chrono::system_clock::now();
         const unsigned int width = runner->Scene->Camera->Settings.Bounds.x;
         const unsigned int height = runner->Scene->Camera->Settings.Bounds.y;

         poly::GPUMemoryManager memory_manager = poly::GPUMemoryManager(width, height);
         size_t bytes_copied = memory_manager.MallocScene(runner->Scene);
         const auto copy_end_time = std::chrono::system_clock::now();
         const std::chrono::duration<double> copy_duration = copy_end_time - copy_start_time;
         
         float effective_bandwidth = ((float) bytes_copied) / (float)copy_duration.count();
         std::string bandwidth_string = convertSize((size_t)effective_bandwidth);
         std::string comma_string = add_commas(bytes_copied);
         Log.debug("Copied " + add_commas(bytes_copied) + " bytes in " + std::to_string(copy_duration.count()) + " (" + bandwidth_string + ").");
         Log.info("Rendering with " + std::to_string(runner->NumSamples) + "spp...");
         
         poly::PathTracerKernel path_tracer_kernel(&memory_manager);
         const auto render_start_time = std::chrono::system_clock::now();
         path_tracer_kernel.Trace(runner->NumSamples);
         const auto render_end_time = std::chrono::system_clock::now();
         const std::chrono::duration<double> render_duration = render_end_time - render_start_time;
         auto foo = render_duration.count();
         Log.info("Sampling complete in %f s.", foo);

         Log.info("Outputting samples to film...");
         const auto output_start_time = std::chrono::system_clock::now();
         poly::OutputPNG::Output(&memory_manager, options.output_filename);
         const auto output_end_time = std::chrono::system_clock::now();
         const std::chrono::duration<double> output_duration = output_end_time - output_start_time;
         Log.info("Output complete in %f s.", output_duration.count());
      }

      const auto totalRunTimeEnd = std::chrono::system_clock::now();
      const std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;

      Log.info("Total computation time: %f", totalElapsedSeconds.count());
      Log.info("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
