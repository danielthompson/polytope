//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <sstream>
#include <map>

#include "../common/utilities/OptionsParser.h"
#include "../common/utilities/Common.h"
#include "integrators/PathTraceIntegrator.h"
#include "samplers/samplers.h"
#include "filters/box_filter.h"
#include "films/png_film.h"
#include "../common/parsers/pbrt_parser.h"
#include "shapes/mesh.h"
#include "structures/stats.h"

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
thread_local poly::random_number_generator rng;

//std::vector<std::thread> threads;

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
Usage: polytope [options] -inputfile <filename> [-outputfile <filename>]

Rendering options:
   -threads <n>      Number of CPU threads to use for rendering. Optional;
                     defaults to the number of detected logical cores.
   -samples <n>      Number of samples to use per pixel. Optional; overrides
                     the number of samples specified in the scene file.

File options:
   -inputfile        The scene file to render. Currently, PBRT is the only
                     supported file format. Optional but strongly encouraged;
                     defaults to a boring example scene.
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

      constexpr unsigned int width = 640;
      constexpr unsigned int height = 480;

      const poly::bounds bounds(width, height);

      const unsigned int concurrentThreadsSupported = std::thread::hardware_concurrency();
      LOG_INFO("Detected " << concurrentThreadsSupported << " cores.");

      unsigned int usingThreads = concurrentThreadsSupported;

      if (options.threadsSpecified && options.threads > 0 && options.threads <= concurrentThreadsSupported) {
         usingThreads = options.threads;
      }

      LOG_INFO("Using " << usingThreads << " threads.");

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
            exit(0);
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
         
         LOG_DEBUG("Image is [" << runner->Bounds.x << "] x [" << runner->Bounds.y << "], " << runner->sample_count << " spp.");
         LOG_DEBUG("Rendering...");

         const auto renderingStart = std::chrono::system_clock::now();

         //   runner->Run();

         std::map<std::thread::id, int> threadMap;
         std::vector<std::thread> threads;
         poly::stats stats;
         
         for (int i = 0; i < usingThreads; i++) {

            LOG_DEBUG("Starting thread " << i << "...");
            threads.emplace_back(runner->spawn_thread(i/*, stats*/));
            const std::thread::id threadID = threads[i].get_id();
            threadMap[threadID] = i;

            // set thread affinity
            // linux only
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            int rc = pthread_setaffinity_np(threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0)
               LOG_DEBUG("Couldn't set thread affinity :/");
         }

         for (int i = 0; i < usingThreads; i++) {
            threads[i].join();
            LOG_DEBUG("Joined thread " << i << ".");
         }

         const auto renderingEnd = std::chrono::system_clock::now();

         const std::chrono::duration<double> renderingElapsedSeconds = renderingEnd - renderingStart;
         LOG_DEBUG("Rendering complete in " << renderingElapsedSeconds.count() << "s.");

         LOG_DEBUG("Outputting to film...");
         const auto outputStart = std::chrono::system_clock::now();
         runner->output();
         const auto outputEnd = std::chrono::system_clock::now();

         const std::chrono::duration<double> outputtingElapsedSeconds = outputEnd - outputStart;
         LOG_DEBUG("Outputting complete in " << outputtingElapsedSeconds.count() << "s.");
      }
      

      const auto totalRunTimeEnd = std::chrono::system_clock::now();
      const std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;

      LOG_INFO("Total computation time: " << totalElapsedSeconds.count() << ".");
      LOG_DEBUG("Camera rays traced: " << main_stats.num_camera_rays);
      LOG_DEBUG("Bounding box intersections: " << main_stats.num_bb_intersections);
      LOG_DEBUG("Bounding box hits (inside): " << main_stats.num_bb_intersections_hit_inside);
      LOG_DEBUG("Bounding box hits (outside): " << main_stats.num_bb_intersections_hit_outside);
      LOG_DEBUG("Bounding box misses: " << main_stats.num_bb_intersections_miss);
      LOG_DEBUG("Bounding box hit %: " << 100.f * (float)(main_stats.num_bb_intersections_hit_inside + main_stats.num_bb_intersections_hit_outside) / (float)main_stats.num_bb_intersections);
      LOG_DEBUG("Triangle intersections: " << main_stats.num_triangle_intersections);
      LOG_DEBUG("Triangle hits: " << main_stats.num_triangle_intersections_hit);
      LOG_DEBUG("Triangle hit %: " << 100.f * (float)main_stats.num_triangle_intersections_hit / (float)main_stats.num_triangle_intersections);
      LOG_INFO("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
