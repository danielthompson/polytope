//
// Created by Daniel Thompson on 3/7/18.
//

#include "DebugIntegrator.h"

namespace Polytope {

   Sample DebugIntegrator::GetSample(Ray &ray, int depth, int x, int y) {
      if (x == 180 && y == 230) {
         x++;
         x--;
      }

      Intersection closestStateToRay = Scene->GetNearestShape(ray, x, y);

      if (closestStateToRay.Hits) {
         ReflectanceSpectrum refl = closestStateToRay.Shape->Material->ReflectanceSpectrum;
         return Sample(SpectralPowerDistribution(refl.r, refl.g, refl.b));
      }

      else {
         return Sample(SpectralPowerDistribution());
      }
   }
}