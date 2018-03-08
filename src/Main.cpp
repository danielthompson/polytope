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
void run();

int main() {

   log("Polytope started.");

   auto start = std::chrono::system_clock::now();

   int x = 640;
   int y = 480;

   using namespace Polytope;

   AbstractSampler *sampler = new CenterSampler();

   SceneBuilder sceneBuilder = SceneBuilder();

   AbstractScene *scene = sceneBuilder.Default(x, y);

   compile(scene);

   AbstractIntegrator *integrator = new DebugIntegrator(scene, 3);

   AbstractFilm *film = new PNGFilm(x, y, "test.png");

   AbstractRunner *runner = new PixelRunner(sampler, scene, integrator, film, x, y);

   runner->Run();

   film->Output();

   delete runner;
   delete film;
   delete integrator;
   delete scene;
   delete sampler;

   auto end = std::chrono::system_clock::now();

   std::chrono::duration<double> elapsed_seconds = end - start;
   std::time_t end_time = std::chrono::system_clock::to_time_t(end);

   std::cout << "finished computation at " << std::ctime(&end_time)
             << "elapsed time: " << elapsed_seconds.count() << "s\n";

   return 0;
}

void compile(Polytope::AbstractScene *scene) {

   unsigned int concurrentThreadsSupported = std::thread::hardware_concurrency();
   log("Detected " + std::to_string(concurrentThreadsSupported) + " cores.");
   log("Using 1 thread until multi-threading is implemented.");

   scene->Compile();
   log("Scene has " + std::to_string(scene->Shapes.size()) + " shapes, " + std::to_string(scene->Lights.size()) + " lights.");
   log("Scene is implemented with " + scene->ImplementationType + ".");


}