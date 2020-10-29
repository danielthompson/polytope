//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <sstream>
#include <map>
#include <atomic>

#include "../common/utilities/OptionsParser.h"
#include "../common/utilities/Common.h"
#include "scenes/SceneBuilder.h"
#include "integrators/PathTraceIntegrator.h"
#include "samplers/samplers.h"
#include "filters/BoxFilter.h"
#include "runners/TileRunner.h"
#include "films/PNGFilm.h"
#include "../common/parsers/PBRTFileParser.h"
#include "../gl/GLRenderer.h"
#include "shapes/mesh.h"
#include "structures/stats.h"

#ifdef __CYGWIN__
#include "platforms/win32-cygwin.h"
#endif

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
   -gl               Render the scene with OpenGL, for reference.
   --help            Print this help text and exit.)");
         std::cout << std::endl;
         exit(0);
      }

      const auto totalRunTimeStart = std::chrono::system_clock::now();

      constexpr unsigned int width = 640;
      constexpr unsigned int height = 480;

      const poly::Bounds bounds(width, height);

      const unsigned int concurrentThreadsSupported = std::thread::hardware_concurrency();
      Log.info("Detected " + std::to_string(concurrentThreadsSupported) + " cores.");

      unsigned int usingThreads = concurrentThreadsSupported;

      if (options.threadsSpecified && options.threads > 0 && options.threads <= concurrentThreadsSupported) {
         usingThreads = options.threads;
      }

      Log.info("Using " + std::to_string(usingThreads) + " threads.");

      {
         std::unique_ptr<poly::AbstractRunner> runner;
         if (options.inputSpecified) {
            // load file
            const auto parse_start = std::chrono::system_clock::now();
            poly::PBRTFileParser parser = poly::PBRTFileParser();
            runner = parser.ParseFile(options.input_filename);
            const auto parse_end = std::chrono::system_clock::now();
            const std::chrono::duration<double> parse_duration = parse_end - parse_start;
            Log.debug("Parsed scene description in " + std::to_string(parse_duration.count()) + "s.");
            
            // override parsed with options here
            if (options.samplesSpecified) {
               runner->NumSamples = options.samples;
            }
         } else {
            Log.debug("No input file specified, using default scene.");
            poly::SceneBuilder sceneBuilder = poly::SceneBuilder(bounds);
            poly::Scene *scene = sceneBuilder.Default();

            // TODO fix
            // Compile(scene);

            std::unique_ptr<poly::AbstractSampler> sampler = std::make_unique<poly::HaltonSampler>();
            std::unique_ptr<poly::AbstractIntegrator> integrator = std::make_unique<poly::PathTraceIntegrator>(scene,
                                                                                                               5);

            std::unique_ptr<poly::BoxFilter> filter = std::make_unique<poly::BoxFilter>(bounds);
            filter->SetSamples(options.samples);
            std::unique_ptr<poly::AbstractFilm> film = std::make_unique<poly::PNGFilm>(bounds, options.output_filename,
                                                                                       std::move(filter));

            runner = std::make_unique<poly::TileRunner>(std::move(sampler), scene, std::move(integrator),
                                                        std::move(film), bounds, options.samples);
         }

         const auto bound_start = std::chrono::system_clock::now();
         thread_stats.num_bvh_bound_leaf_same_centroid = 0;
         unsigned int num_nodes = runner->Scene->bvh_root.bound(runner->Scene->Shapes);
         const auto bound_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> bound_duration = bound_end - bound_start;
         Log.debug("Created BVH in " + std::to_string(bound_duration.count()) + "s.");
         Log.debug("Number of leaves with multiple triangles with the same centroid: " + std::to_string(thread_stats.num_bvh_bound_leaf_same_centroid));
         
         const auto compact_start = std::chrono::system_clock::now();
         runner->Scene->bvh_root.compact();
         const auto compact_end = std::chrono::system_clock::now();
         const std::chrono::duration<double> compact_duration = compact_end - compact_start;
         Log.debug("Compacted BVH in " + std::to_string(compact_duration.count()) + "s.");
         
         runner->Scene->bvh_root.metrics();
         
         Log.debug(
               std::string("Image is [") +
               std::to_string(runner->Bounds.x) +
               std::string("] x [") +
               std::to_string(runner->Bounds.y) +
               std::string("], ") +
               std::to_string(runner->NumSamples) + " spp.");

         if (options.gl) {
            Log.debug("Rasterizing with OpenGL...");
            poly::GLRenderer renderer;
            renderer.Render(runner->Scene);
         }
         else {
            Log.debug("Rendering...");

            const auto renderingStart = std::chrono::system_clock::now();

            //   runner->Run();

            std::map<std::thread::id, int> threadMap;
            std::vector<std::thread> threads;
            poly::stats stats;
            
            for (int i = 0; i < usingThreads; i++) {

               Log.debug(std::string("Starting thread " + std::to_string(i) + std::string("...")));
               threads.emplace_back(runner->Spawn(i/*, stats*/));
               const std::thread::id threadID = threads[i].get_id();
               threadMap[threadID] = i;

               // set thread affinity
               // linux only
               cpu_set_t cpuset;
               CPU_ZERO(&cpuset);
               CPU_SET(i, &cpuset);
               int rc = pthread_setaffinity_np(threads[i].native_handle(), sizeof(cpu_set_t), &cpuset);
               if (rc != 0)
                  Log.debug("Couldn't set thread affinity :/");
            }

            for (int i = 0; i < usingThreads; i++) {
               threads[i].join();
               Log.debug(std::string("Joined thread " + std::to_string(i) + std::string(".")));
            }

            const auto renderingEnd = std::chrono::system_clock::now();

            const std::chrono::duration<double> renderingElapsedSeconds = renderingEnd - renderingStart;
            Log.debug("Rendering complete in " + std::to_string(renderingElapsedSeconds.count()) + "s.");

            Log.debug("Outputting to film...");
            const auto outputStart = std::chrono::system_clock::now();
            runner->Output();
            const auto outputEnd = std::chrono::system_clock::now();

            const std::chrono::duration<double> outputtingElapsedSeconds = outputEnd - outputStart;
            Log.debug("Outputting complete in " + std::to_string(outputtingElapsedSeconds.count()) + "s.");
         }
      }

      const auto totalRunTimeEnd = std::chrono::system_clock::now();
      const std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;

      Log.info("Total computation time: " + std::to_string(totalElapsedSeconds.count()) + ".");
      Log.debug("Camera rays traced: " + std::to_string(main_stats.num_camera_rays));
      Log.debug("Bounding box intersections: " + std::to_string(main_stats.num_bb_intersections));
      Log.debug("Bounding box hits (inside): " + std::to_string(main_stats.num_bb_intersections_hit_inside));
      Log.debug("Bounding box hits (outside): " + std::to_string(main_stats.num_bb_intersections_hit_outside));
      Log.debug("Bounding box misses: " + std::to_string(main_stats.num_bb_intersections_miss));
      Log.debug("Bounding box hit %: " + std::to_string(100.f * (float)(main_stats.num_bb_intersections_hit_inside + main_stats.num_bb_intersections_hit_outside) / (float)main_stats.num_bb_intersections));
      Log.debug("Triangle intersections: " + std::to_string(main_stats.num_triangle_intersections));
      Log.debug("Triangle hits: " + std::to_string(main_stats.num_triangle_intersections_hit));
      Log.debug("Triangle hit %: " + std::to_string(100.f * (float)main_stats.num_triangle_intersections_hit / (float)main_stats.num_triangle_intersections));
      Log.info("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
