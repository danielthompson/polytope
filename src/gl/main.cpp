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

void segfaultHandler(int signalNumber) {
   ERROR("Segfault (signal %i). Stacktrace to be implemented...", signalNumber);
}

void signalHandler(int signalNumber) {
   ERROR("Interrupt (signal %i). Stacktrace to be implemented...", signalNumber);
}

bool hasAbortedOnce = false;

void userAbortHandler(int signalNumber) {
   if (hasAbortedOnce) {
      Log.info("Aborting at user request.");
      exit(signalNumber);
   }
   else {
      Log.info("Detected Ctrl-C keypress. Ignoring since it's the first time. Press Ctrl-C again to really quit.");
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
         std::shared_ptr<poly::AbstractRunner> runner;
         if (options.inputSpecified) {
            // load file
            const auto parse_start = std::chrono::system_clock::now();
            poly::pbrt_parser parser = poly::pbrt_parser();
            runner = parser.parse_file(options.input_filename);
            const auto parse_end = std::chrono::system_clock::now();
            const std::chrono::duration<double> parse_duration = parse_end - parse_start;
            Log.debug("Parsed scene description in " + std::to_string(parse_duration.count()) + "s.");
            
            // override parsed with options here
            if (options.samplesSpecified) {
               runner->NumSamples = options.samples;
            }
         } else {
            Log.debug("No input file specified, quitting.");
            exit(0);
         }

         const auto bound_start = std::chrono::system_clock::now();
         thread_stats.num_bvh_bound_leaf_same_centroid = 0;
         unsigned int num_nodes = runner->Scene->bvh_root.bound(runner->Scene->Shapes);
         const auto bound_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> bound_duration = bound_end - bound_start;
         Log.debug("Created BVH with " + std::to_string(num_nodes) + " nodes in " + std::to_string(bound_duration.count()) + "s.");
         Log.debug("Number of leaves with multiple triangles with the same centroid: " + std::to_string(thread_stats.num_bvh_bound_leaf_same_centroid));
         
         const auto compact_start = std::chrono::system_clock::now();
         runner->Scene->bvh_root.compact();
         const auto compact_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> compact_duration = compact_end - compact_start;
         Log.debug("Compacted BVH in " + std::to_string(compact_duration.count()) + "s.");
         
         runner->Scene->bvh_root.metrics();

         Log.debug("Rasterizing with OpenGL...");
         poly::gl_renderer renderer(runner);
         renderer.render();
         
      }

      Log.info("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
