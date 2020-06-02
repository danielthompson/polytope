//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <sstream>
#include <map>

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

#ifdef __CYGWIN__
#include "platforms/win32-cygwin.h"
#endif

Polytope::Logger Log;

void segfaultHandler(int signalNumber) {
   Log.WithTime("Detected a segfault. Stacktrace to be implemented...");
#ifdef __CYGWIN__
   //printStack();
#endif
   exit(signalNumber);
}

void signalHandler(int signalNumber) {
   std::ostringstream oss;
   oss << "Received interrupt signal " << signalNumber << ", aborting.";
   Log.WithTime(oss.str());
   exit(signalNumber);
}

bool hasAbortedOnce = false;

void userAbortHandler(int signalNumber) {
   if (hasAbortedOnce) {
      Log.WithTime("Aborting at user request.");
      exit(signalNumber);
   }
   else {
      Log.WithTime("Detected Ctrl-C keypress. Ignoring since it's the first time. Press Ctrl-C again to really quit.");
      hasAbortedOnce = true;
   }
}

int main(int argc, char* argv[]) {

   try {
      Log = Polytope::Logger();

      Polytope::Options options = Polytope::Options();

      if (argc > 0) {
         Polytope::OptionsParser parser(argc, argv);
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

      const Polytope::Bounds bounds(width, height);

      const unsigned int concurrentThreadsSupported = std::thread::hardware_concurrency();
      Log.WithTime("Detected " + std::to_string(concurrentThreadsSupported) + " cores.");

      unsigned int usingThreads = concurrentThreadsSupported;

      if (options.threadsSpecified && options.threads > 0 && options.threads <= concurrentThreadsSupported) {
         usingThreads = options.threads;
      }

      Log.WithTime("Using " + std::to_string(usingThreads) + " threads.");

      {
         std::unique_ptr<Polytope::AbstractRunner> runner;
         if (!options.inputSpecified) {
            Log.WithTime("No input file specified, using default scene.");
            Polytope::SceneBuilder sceneBuilder = Polytope::SceneBuilder(bounds);
            Polytope::AbstractScene *scene = sceneBuilder.Default();

            // TODO fix
            // Compile(scene);

            std::unique_ptr<Polytope::AbstractSampler> sampler = std::make_unique<Polytope::HaltonSampler>();
            std::unique_ptr<Polytope::AbstractIntegrator> integrator = std::make_unique<Polytope::PathTraceIntegrator>(scene, 5);

            std::unique_ptr<Polytope::BoxFilter> filter = std::make_unique<Polytope::BoxFilter>(bounds);
            filter->SetSamples(options.samples);
            std::unique_ptr<Polytope::AbstractFilm> film = std::make_unique<Polytope::PNGFilm>(bounds, options.output_filename, std::move(filter));

            runner = std::make_unique<Polytope::TileRunner>(std::move(sampler), scene, std::move(integrator), std::move(film), bounds, options.samples);

         }
         else {
            // load file
            Polytope::PBRTFileParser parser = Polytope::PBRTFileParser();
            runner = parser.ParseFile(options.input_filename);

            // override parsed with options here
            if (options.samplesSpecified) {
               runner->NumSamples = options.samples;
            }
         }

         Log.WithTime(
               std::string("Image is [") +
               std::to_string(runner->Bounds.x) +
               std::string("] x [") +
               std::to_string(runner->Bounds.y) +
               std::string("], ") +
               std::to_string(runner->NumSamples) + " spp.");

         if (options.gl) {
            Log.WithTime("Rasterizing with OpenGL...");
            Polytope::GLRenderer renderer;
            renderer.Render(runner->Scene);
         }
         else {
            Log.WithTime("Rendering...");

            const auto renderingStart = std::chrono::system_clock::now();

            //   runner->Run();

            std::map<std::thread::id, int> threadMap;
            std::vector<std::thread> threads;
            for (int i = 0; i < usingThreads; i++) {

               Log.WithTime(std::string("Starting thread " + std::to_string(i) + std::string("...")));
               threads.emplace_back(runner->Spawn(i));
               const std::thread::id threadID = threads[i].get_id();
               threadMap[threadID] = i;

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