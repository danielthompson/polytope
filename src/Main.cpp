//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <memory>
#include <iomanip>
#include <thread>
#include <sstream>
#include "runners/PixelRunner.h"
#include "samplers/CenterSampler.h"
#include "scenes/AbstractScene.h"
#include "scenes/NaiveScene.h"
#include "scenes/SceneBuilder.h"
#include "integrators/AbstractIntegrator.h"
#include "integrators/PathTraceIntegrator.h"
#include "films/PNGFilm.h"
#include "integrators/DebugIntegrator.h"
#include "filters/BoxFilter.h"
#include "samplers/GridSampler.h"


std::string time_in_HH_MM_SS_MMM()
{
   using namespace std::chrono;

   // get current time
   auto now = system_clock::now();

   // get number of milliseconds for the current second
   // (remainder after division into seconds)
   auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

   // convert to std::time_t in order to convert to std::tm (broken time)
   auto timer = system_clock::to_time_t(now);

   // convert to broken time
   std::tm bt = *std::localtime(&timer);

   std::ostringstream oss;

   oss << std::put_time(&bt, "%T"); // HH:MM:SS
   oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

   return oss.str();
}

void log(std::string text) {
   auto time = std::time(nullptr);
   std::cout << "[" << std::put_time(std::localtime(&time), "%F ") << time_in_HH_MM_SS_MMM() << "] "; // ISO 8601 format.
   std::cout << text << std::endl;
}


void compile(Polytope::AbstractScene *scene);

int main() {

   log("Polytope started.");

   auto totalRunTimeStart = std::chrono::system_clock::now();

   using namespace Polytope;

   constexpr unsigned int numSamples = 4;
   constexpr unsigned int width = 640;
   constexpr unsigned int height = 480;

   const Polytope::Bounds bounds(width, height);

   AbstractSampler *sampler = new GridSampler();

   SceneBuilder sceneBuilder = SceneBuilder(bounds);

   AbstractScene *scene = sceneBuilder.Default();

   compile(scene);

   AbstractIntegrator *integrator = new PathTraceIntegrator(scene, 3);

   AbstractFilm *film = new PNGFilm(bounds, "test.png", std::make_unique<BoxFilter>(bounds));

   AbstractRunner *runner = new PixelRunner(sampler, scene, integrator, film, bounds, numSamples);

   log("Rendering...");
   auto renderingStart = std::chrono::system_clock::now();
   runner->Run();
   auto renderingEnd = std::chrono::system_clock::now();

   std::chrono::duration<double> renderingElapsedSeconds = renderingEnd - renderingStart;
   log("Rendering complete in " + std::to_string(renderingElapsedSeconds.count()) + "s.");

   log("Outputting to film...");
   auto outputStart = std::chrono::system_clock::now();
   film->Output();
   auto outputEnd = std::chrono::system_clock::now();

   std::chrono::duration<double> outputtingElapsedSeconds = outputEnd - outputStart;
   log("Outputting complete in " + std::to_string(outputtingElapsedSeconds.count()) + "s.");

   delete runner;
   delete film;
   delete integrator;
   delete scene;
   delete sampler;

   auto totalRunTimeEnd = std::chrono::system_clock::now();

   std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;
   std::time_t end_time = std::chrono::system_clock::to_time_t(totalRunTimeEnd);

   log("Total computation time: " + std::to_string(totalElapsedSeconds.count()) + ".");

   log("Exiting Polytope.");

   return 0;
}

void compile(Polytope::AbstractScene *scene) {

   unsigned int concurrentThreadsSupported = std::thread::hardware_concurrency();
   log("Detected " + std::to_string(concurrentThreadsSupported) + " cores.");
   log("Using 1 thread until multi-threading is implemented.");
   log("Compiling scene...");

   auto start = std::chrono::system_clock::now();
   scene->Compile();
   auto end = std::chrono::system_clock::now();

   std::chrono::duration<double> elapsed_seconds = end - start;
   log("Compilation complete in " + std::to_string(elapsed_seconds.count()) + "s.");
   log("Scene has " + std::to_string(scene->Shapes.size()) + " shapes, " + std::to_string(scene->Lights.size()) + " lights.");
   log("Scene is implemented with " + scene->ImplementationType + ".");

}