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
#include "../common/parsers/PBRTFileParser.h"
#include "kernels/generate_rays.cuh"
#include "gpu_memory_manager.h"
#include "kernels/path_tracer.cuh"
#include "png_output.h"
#include "mesh/cuda_mesh_soa.h"
#include "../cpu/structures/stats.h"

poly::Logger Log;
struct poly::stats main_stats;
thread_local struct poly::stats thread_stats;

std::atomic<int> num_bb_intersections;
std::atomic<int> num_bb_intersections_origin_inside;
std::atomic<int> num_triangle_intersections;

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

      Log.WithTime("Using 1 CPU thread.");

      {
         if (!options.inputSpecified) {
            Log.WithTime("No input file specified. Quitting.");
            exit(1);
         }
         
         // load file
         poly::PBRTFileParser parser = poly::PBRTFileParser();
         const auto runner = parser.ParseFile(options.input_filename);
         
         // override parsed with options here
         if (options.samplesSpecified) {
            runner->NumSamples = options.samples;
         }
         
         Log.WithTime(
               std::string("Image is [") +
               std::to_string(runner->Bounds.x) +
               std::string("] x [") +
               std::to_string(runner->Bounds.y) +
               std::string("], ") +
               std::to_string(runner->NumSamples) + " spp.");
         
         Log.WithTime("Copying data to GPU...");

         const auto copy_start_time = std::chrono::system_clock::now();
         const unsigned int width = runner->Scene->Camera->Settings.Bounds.x;
         const unsigned int height = runner->Scene->Camera->Settings.Bounds.y;

         poly::GPUMemoryManager memory_manager = poly::GPUMemoryManager(width, height);
         size_t bytes_copied = memory_manager.MallocScene(runner->Scene);
         const auto copy_end_time = std::chrono::system_clock::now();
         const std::chrono::duration<double> copy_duration = copy_end_time - copy_start_time;
         
         float effective_bandwidth = ((float) bytes_copied) / (float)copy_duration.count();
         std::string bandwidth_string = convertSize((size_t)effective_bandwidth);
         Log.WithTime("Copied " + std::to_string(bytes_copied) + " bytes in " + std::to_string(copy_duration.count()) + "s (" + bandwidth_string + ").");
         Log.WithTime("Rendering...");
         
         poly::PathTracerKernel path_tracer_kernel(&memory_manager);
         const auto render_start_time = std::chrono::system_clock::now();
         path_tracer_kernel.Trace();
         const auto render_end_time = std::chrono::system_clock::now();
         const std::chrono::duration<double> render_duration = render_end_time - render_start_time;
         Log.WithTime("Render complete in " + std::to_string(render_duration.count()) + "s.");

         Log.WithTime("Outputting to film...");
         const auto output_start_time = std::chrono::system_clock::now();
         poly::OutputPNG output;
         output.Output(&memory_manager);
         const auto output_end_time = std::chrono::system_clock::now();

         const std::chrono::duration<double> output_duration = output_end_time - output_start_time;
         Log.WithTime("Output complete in " + std::to_string(output_duration.count()) + "s.");
      }

      const auto totalRunTimeEnd = std::chrono::system_clock::now();
      const std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;

      Log.WithTime("Total computation time: " + std::to_string(totalElapsedSeconds.count()) + ".");
      Log.WithTime("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
