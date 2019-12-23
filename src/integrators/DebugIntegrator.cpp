//
// Created by Daniel Thompson on 3/7/18.
//

#include "DebugIntegrator.h"

namespace Polytope {

   Sample DebugIntegrator::GetSample(Ray &ray, const int depth, const int x, const int y) {

      const Intersection closestStateToRay = Scene->GetNearestShape(ray, x, y);

      if (closestStateToRay.Hits) {
         const ReflectanceSpectrum refl = closestStateToRay.Shape->Material->ReflectanceSpectrum;
         return Sample(SpectralPowerDistribution(refl.r, refl.g, refl.b));
      }

      else {
         return Sample(SpectralPowerDistribution());
      }
   }
}