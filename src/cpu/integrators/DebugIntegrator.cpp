//
// Created by Daniel Thompson on 3/7/18.
//

#include "DebugIntegrator.h"

namespace poly {

   poly::Sample DebugIntegrator::get_sample(ray &ray, int depth, int x, int y) {

      const poly::intersection closest_intersection = Scene->intersect(ray, x, y);

      if (closest_intersection.Hits) {
         // TODO
//         const ReflectanceSpectrum refl = closestStateToRay.Shape->Material->ReflectanceSpectrum;
//         return Sample(SpectralPowerDistribution(refl.r, refl.g, refl.b));
         return poly::Sample(poly::SpectralPowerDistribution(ray.min_t, ray.min_t, ray.min_t));
      }

      else {
         return poly::Sample(poly::SpectralPowerDistribution());
      }
   }
}