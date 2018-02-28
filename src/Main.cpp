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
#include "shading/Spectrum.h"

int main() {

   int x = 640;
   int y = 480;

   Polytope::AbstractSampler sampler = Polytope::CenterSampler();

   Polytope::AbstractScene *scene = Polytope::SceneBuilder::Default(x, y);

   Polytope::PixelRunner runner = Polytope::PixelRunner(&sampler, x, y);

   runner.Run();

   std::cout << "Hello, World!" << std::endl;

   delete scene;
   delete sampler;


   return 0;
}
