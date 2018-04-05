//
// Created by Daniel on 21-Mar-18.
//

#include <thread>
#include <iostream>
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

namespace Polytope {

   void Tracer::Run() {

      auto totalRunTimeStart = std::chrono::system_clock::now();

      constexpr unsigned int width = 640;
      constexpr unsigned int height = 640;

      const Polytope::Bounds bounds(width, height);

      SceneBuilder sceneBuilder = SceneBuilder(bounds);

      AbstractScene *scene = sceneBuilder.Default();

      Compile(scene);

      const unsigned int concurrentThreadsSupported = std::thread::hardware_concurrency();
      Logger.LogTime("Detected " + std::to_string(concurrentThreadsSupported) + " cores.");

      unsigned int usingThreads = concurrentThreadsSupported;

      if (Options.threads > 0 && Options.threads <= concurrentThreadsSupported) {
         usingThreads = Options.threads;
      }

      Logger.LogTime("Using " + std::to_string(usingThreads) + " threads.");

      {
         std::unique_ptr<AbstractSampler> sampler = std::make_unique<HaltonSampler>();
         std::unique_ptr<AbstractIntegrator> integrator = std::make_unique<PathTraceIntegrator>(scene, 3);
         std::unique_ptr<AbstractFilm> film = std::make_unique<PNGFilm>(bounds, Options.filename, std::make_unique<BoxFilter>(bounds, Options.samples));
         const std::unique_ptr<AbstractRunner> runner = std::make_unique<TileRunner>(std::move(sampler), scene, std::move(integrator), std::move(film), bounds, Options.samples);

         Logger.LogTime(
               std::string("Image is [") + std::to_string(width) + std::string("] x [") + std::to_string(height) +
               std::string("], ") + std::to_string(Options.samples) + " spp.");

         Logger.LogTime("Rendering...");

         auto renderingStart = std::chrono::system_clock::now();

         //   runner->Run();

         std::map<std::thread::id, int> threadMap;
         std::vector<std::thread> threads;
         for (int i = 0; i < usingThreads; i++) {

            threads.emplace_back(runner->Spawn());
            const std::thread::id threadID = threads[i].get_id();
            threadMap[threadID] = i;
            Logger.LogTime(std::string("Started thread " + std::to_string(i) + std::string(".")));
         }

         for (int i = 0; i < usingThreads; i++) {
            threads[i].join();
            Logger.LogTime(std::string("Joined thread " + std::to_string(i) + std::string(".")));
         }

         auto renderingEnd = std::chrono::system_clock::now();

         std::chrono::duration<double> renderingElapsedSeconds = renderingEnd - renderingStart;
         Logger.LogTime("Rendering complete in " + std::to_string(renderingElapsedSeconds.count()) + "s.");

         Logger.LogTime("Outputting to film...");
         auto outputStart = std::chrono::system_clock::now();
         runner->Output();
         auto outputEnd = std::chrono::system_clock::now();

         std::chrono::duration<double> outputtingElapsedSeconds = outputEnd - outputStart;
         Logger.LogTime("Outputting complete in " + std::to_string(outputtingElapsedSeconds.count()) + "s.");

      }

      delete scene;

      auto totalRunTimeEnd = std::chrono::system_clock::now();

      std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;
      std::time_t end_time = std::chrono::system_clock::to_time_t(totalRunTimeEnd);

      Logger.LogTime("Total computation time: " + std::to_string(totalElapsedSeconds.count()) + ".");

      Logger.LogTime("Exiting Polytope.");

   }

   void Tracer::Compile(AbstractScene *scene) {

      Logger.LogTime("Compiling scene...");

      auto start = std::chrono::system_clock::now();
      scene->Compile();
      auto end = std::chrono::system_clock::now();

      std::chrono::duration<double> elapsed_seconds = end - start;
      Logger.LogTime("Compilation complete in " + std::to_string(elapsed_seconds.count()) + "s.");
      Logger.LogTime(
            "Scene has " + std::to_string(scene->Shapes.size()) + " shapes, " + std::to_string(scene->Lights.size()) +
            " lights.");
      Logger.LogTime("Scene is implemented with " + scene->ImplementationType + ".");

   }
}
