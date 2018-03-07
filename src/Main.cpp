//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <memory>
#include "runners/PixelRunner.h"
#include "samplers/CenterSampler.h"
#include "scenes/AbstractScene.h"
#include "scenes/NaiveScene.h"
#include "scenes/SceneBuilder.h"
#include "integrators/AbstractIntegrator.h"
#include "integrators/PathTraceIntegrator.h"
#include "films/PNGFilm.h"
#include "integrators/DebugIntegrator.h"


int main() {

   int x = 640;
   int y = 480;

   using namespace Polytope;

   AbstractSampler *sampler = new CenterSampler();

   SceneBuilder sceneBuilder = SceneBuilder();

   AbstractScene *scene = sceneBuilder.Default(x, y);

   AbstractIntegrator *integrator = new DebugIntegrator(scene, 3);

   AbstractFilm *film = new PNGFilm(x, y, "test.png");

   AbstractRunner *runner = new PixelRunner(sampler, scene, integrator, film, x, y);

   runner->Run();

   film->Output();

   std::cout << "Hello, World!" << std::endl;

   delete runner;
   delete film;
   delete integrator;
   delete scene;
   delete sampler;

   return 0;
}
