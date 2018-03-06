//
// Created by Daniel Thompson on 2/21/18.
//


#include "AbstractRunner.h"
#include "../structures/Ray.h"

namespace Polytope {

   void AbstractRunner::Trace(int x, int y) {
      Point2f sampleLocation = Sampler->GetSample(x, y);
      Ray ray = Scene->Camera->GetRay(sampleLocation);
      Sample sample = Integrator->GetSample(ray, 0, x, y);
      Film->AddSample(sampleLocation, sample);
   }



}