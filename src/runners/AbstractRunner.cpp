//
// Created by Daniel Thompson on 2/21/18.
//

#include "AbstractRunner.h"
#include "../structures/Ray.h"

namespace Polytope {

   void Polytope::AbstractRunner::Trace(int x, int y) {

      Point2 sample = Sampler->GetSample(x, y);

      //Ray cameraRay =
   }

   AbstractRunner::AbstractRunner(AbstractSampler *sampler)
         : Sampler(sampler) { }

}