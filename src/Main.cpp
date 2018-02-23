//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <memory>
#include "runners/PixelRunner.h"
#include "samplers/CenterSampler.h"

int main() {

   int x = 640;
   int y = 480;

   Polytope::AbstractSampler *sampler = new Polytope::CenterSampler() ;

   Polytope::PixelRunner runner = Polytope::PixelRunner(sampler, x, y);

   runner.Run();

   std::cout << "Hello, World!" << std::endl;

   delete sampler;

   return 0;
}
