//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <sstream>

#include "../common/utilities/OptionsParser.h"
#include "../common/utilities/Common.h"
#include "../cpu/integrators/PathTraceIntegrator.h"
#include "../cpu/samplers/samplers.h"
#include "../cpu/filters/BoxFilter.h"
#include "../common/parsers/pbrt_parser.h"
#include "gl_renderer.h"

poly::Logger Log;

void segfault_handler(int signal_number) {
   ERROR("Segfault (signal " << signal_number << "). Stacktrace to be implemented...");
}

void signal_handler(int signal_number) {
   ERROR("Interrupt (signal " << signal_number << "). Stacktrace to be implemented...");
}

bool hasAbortedOnce = false;

void userAbortHandler(int signalNumber) {
   if (hasAbortedOnce) {
      LOG_INFO("Aborting at user request.");
      exit(signalNumber);
   }
   else {
      LOG_INFO("Detected Ctrl-C keypress. Ignoring since it's the first time. Press Ctrl-C again to really quit.");
      hasAbortedOnce = true;
   }
}

struct poly::stats main_stats;
thread_local struct poly::stats thread_stats;

int main(int argc, char* argv[]) {

   main_stats.num_triangle_intersections = 0;
   main_stats.num_bb_intersections = 0;
   
   try {
      Log = poly::Logger();

      poly::Options options = poly::Options();

      if (argc > 0) {
         poly::OptionsParser parser(argc, argv);
         options = parser.Parse();
      }

      if (options.help) {
         std::cout << "Polytope by Daniel A. Thompson, built on " << __DATE__ << std::endl;
         fprintf(stderr, R"(
Usage: polytope-gl -inputfile <filename>

File options:
   -inputfile        The scene file to render. Currently, PBRT is the only
                     supported file format. Optional but strongly encouraged;
                     defaults to a boring example scene.

Other:
   --help            Print this help text and exit.)");
         std::cout << std::endl;
         exit(0);
      }

      {
         std::shared_ptr<poly::runner> runner;
         if (options.inputSpecified) {
            // load file
            const auto parse_start = std::chrono::system_clock::now();
            poly::pbrt_parser parser = poly::pbrt_parser();
            runner = parser.parse_file(options.input_filename);
            const auto parse_end = std::chrono::system_clock::now();
            const std::chrono::duration<double> parse_duration = parse_end - parse_start;
            LOG_DEBUG("Parsed scene description in " << parse_duration.count() << "s.");
            
            // override parsed with options here
            if (options.samplesSpecified) {
               runner->sample_count = options.samples;
            }
         } else {
            LOG_ERROR("No input file specified; exiting.");
            exit(EXIT_FAILURE);
         }

         const auto bound_start = std::chrono::system_clock::now();
         thread_stats.num_bvh_bound_leaf_same_centroid = 0;
         unsigned int num_nodes = runner->Scene->bvh_root.bound(runner->Scene->Shapes);
         const auto bound_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> bound_duration = bound_end - bound_start;
         LOG_DEBUG("Created BVH with " << num_nodes << " nodes in " << bound_duration.count() << "s.");
         LOG_DEBUG("Number of leaves with multiple triangles with the same centroid: " << thread_stats.num_bvh_bound_leaf_same_centroid);
         
         const auto compact_start = std::chrono::system_clock::now();
         runner->Scene->bvh_root.compact();
         const auto compact_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> compact_duration = compact_end - compact_start;
         LOG_DEBUG("Compacted BVH in " << compact_duration.count() << "s.");
         
         runner->Scene->bvh_root.metrics();

         LOG_DEBUG("Rasterizing with OpenGL...");
         poly::gl_renderer renderer(runner);
         renderer.render();
         
      }

      LOG_INFO("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
