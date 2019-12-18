//
// Created by Daniel on 21-Mar-18.
//

#include <thread>
#include <map>
#include "Tracer.h"
#include "samplers/AbstractSampler.h"
#include "samplers/GridSampler.h"
#include "scenes/SceneBuilder.h"
#include "integrators/AbstractIntegrator.h"
#include "films/AbstractFilm.h"
#include "integrators/PathTraceIntegrator.h"
#include "films/PNGFilm.h"
#include "filters/BoxFilter.h"
#include "runners/TileRunner.h"
#include "samplers/HaltonSampler.h"
#include "integrators/DebugIntegrator.h"
#include "parsers/PBRTFileParser.h"
#include "utilities/Common.h"


namespace Polytope {

   void Tracer::Run() {

      const auto totalRunTimeStart = std::chrono::system_clock::now();

      constexpr unsigned int width = 640;
      constexpr unsigned int height = 480;

      const Polytope::Bounds bounds(width, height);

      const unsigned int concurrentThreadsSupported = std::thread::hardware_concurrency();
      Log.WithTime("Detected " + std::to_string(concurrentThreadsSupported) + " cores.");

      unsigned int usingThreads = concurrentThreadsSupported;

      if (Options.threadsSpecified && Options.threads > 0 && Options.threads <= concurrentThreadsSupported) {
         usingThreads = Options.threads;
      }

      Log.WithTime("Using " + std::to_string(usingThreads) + " threads.");

      {
         std::unique_ptr<AbstractRunner> runner;
         if (!Options.inputSpecified) {
            Log.WithTime("No input file specified, using default scene.");
            SceneBuilder sceneBuilder = SceneBuilder(bounds);
            AbstractScene *scene = sceneBuilder.Default();
            Compile(scene);

            std::unique_ptr<AbstractSampler> sampler = std::make_unique<HaltonSampler>();
            std::unique_ptr<AbstractIntegrator> integrator = std::make_unique<PathTraceIntegrator>(scene, 5);

            std::unique_ptr<BoxFilter> filter = std::make_unique<BoxFilter>(bounds);
            filter->SetSamples(Options.samples);
            std::unique_ptr<AbstractFilm> film = std::make_unique<PNGFilm>(bounds, Options.output_filename, std::move(filter));

            runner = std::make_unique<TileRunner>(std::move(sampler), scene, std::move(integrator), std::move(film), bounds, Options.samples);

         }
         else {
            // load file
            PBRTFileParser parser = PBRTFileParser();
            runner = parser.ParseFile(Options.input_filename);

            // override parsed with options here
            if (Options.samplesSpecified) {
               runner->NumSamples = Options.samples;
            }
         }

         Log.WithTime(
               std::string("Image is [") + std::to_string(width) + std::string("] x [") + std::to_string(height) +
               std::string("], ") + std::to_string(runner->NumSamples) + " spp.");

         Log.WithTime("Rendering...");

         const auto renderingStart = std::chrono::system_clock::now();

         //   runner->Run();

         std::map<std::thread::id, int> threadMap;
         std::vector<std::thread> threads;
         for (int i = 0; i < usingThreads; i++) {

            threads.emplace_back(runner->Spawn());
            const std::thread::id threadID = threads[i].get_id();
            threadMap[threadID] = i;
            Log.WithTime(std::string("Started thread " + std::to_string(i) + std::string(".")));
         }

         for (int i = 0; i < usingThreads; i++) {
            threads[i].join();
            Log.WithTime(std::string("Joined thread " + std::to_string(i) + std::string(".")));
         }

         const auto renderingEnd = std::chrono::system_clock::now();

         const std::chrono::duration<double> renderingElapsedSeconds = renderingEnd - renderingStart;
         Log.WithTime("Rendering complete in " + std::to_string(renderingElapsedSeconds.count()) + "s.");

         Log.WithTime("Outputting to film...");
         const auto outputStart = std::chrono::system_clock::now();
         runner->Output();
         const auto outputEnd = std::chrono::system_clock::now();

         const std::chrono::duration<double> outputtingElapsedSeconds = outputEnd - outputStart;
         Log.WithTime("Outputting complete in " + std::to_string(outputtingElapsedSeconds.count()) + "s.");
      }

      const auto totalRunTimeEnd = std::chrono::system_clock::now();
      const std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;

      Log.WithTime("Total computation time: " + std::to_string(totalElapsedSeconds.count()) + ".");
   }

   void Tracer::Compile(AbstractScene *scene) {

      Log.WithTime("Compiling scene...");

      const auto start = std::chrono::system_clock::now();
      scene->Compile();
      const auto end = std::chrono::system_clock::now();

      std::chrono::duration<double> elapsed_seconds = end - start;
      Log.WithTime("Compilation complete in " + std::to_string(elapsed_seconds.count()) + "s.");
      Log.WithTime(
            "_scene has " + std::to_string(scene->Shapes.size()) + " shapes, " + std::to_string(scene->Lights.size()) +
            " lights.");
      Log.WithTime("_scene is implemented with " + scene->ImplementationType + ".");
   }
}
