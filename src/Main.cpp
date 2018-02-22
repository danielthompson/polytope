//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include "runners/PixelRunner.h"
#include "samplers/AbstractSampler.h"
#include "samplers/CenterSampler.h"

int main() {

   int x = 640;
   int y = 480;

   Polytope::AbstractSampler sampler = Polytope::CenterSampler();

   Polytope::PixelRunner runner = Polytope::PixelRunner(x, y);

   runner.Run();


   std::cout << "Hello, World!" << std::endl;
   return 0;
}
