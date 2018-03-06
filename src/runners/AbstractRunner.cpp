//
// Created by Daniel Thompson on 2/21/18.
//


#include "AbstractRunner.h"
#include "../structures/Ray.h"

namespace Polytope {

   void AbstractRunner::Trace(int x, int y) {

      Point2f sample = Sampler->GetSample(x, y);
      Ray ray = Scene->Camera->GetRay(sample);
      Integrator->GetSample(ray, 0, x, y);

      //Ray cameraRay =
   }



}