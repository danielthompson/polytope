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


int main() {

   int x = 640;
   int y = 480;

   using namespace Polytope;

   AbstractSampler sampler = CenterSampler();

   SceneBuilder sceneBuilder = SceneBuilder();

   AbstractScene *scene = sceneBuilder.Default(x, y);

   PixelRunner runner = PixelRunner(&sampler, x, y);

   runner.Run();

   std::cout << "Hello, World!" << std::endl;

   delete scene;
   delete sampler;


   return 0;
}
