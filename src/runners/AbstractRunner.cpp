//
// Created by Daniel Thompson on 2/21/18.
//


#include <iostream>
#include "AbstractRunner.h"
#include "../structures/Ray.h"

namespace Polytope {

   void AbstractRunner::Trace(int x, int y) {
      Point2f sampleLocation = Sampler->GetSample(x, y);
      Ray ray = Scene->Camera->GetRay(sampleLocation);

      if (x == 0 && y == 113) {
         x++;
         x--;
      }

      Sample sample = Integrator->GetSample(ray, 0, x, y);
      Film->AddSample(sampleLocation, sample);
   }
}