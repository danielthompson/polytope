//
// Created by Daniel Thompson on 3/5/18.
//

#include <iostream>
#include "PathTraceIntegrator.h"

namespace Polytope {

   Sample PathTraceIntegrator::GetSample(Ray &ray, int depth, int x, int y) {

      Intersection intersection = Scene->GetNearestShape(ray, x, y);

      if (!intersection.Hits) {
         return Sample(SpectralPowerDistribution());
      }

      if (intersection.Shape->IsLight()) {
         return Sample(intersection.Shape->Light->SpectralPowerDistribution);
      }

      // base case
      if (depth >= MaxDepth) {
         return Sample(SpectralPowerDistribution());
      }
      else {
         std::shared_ptr<AbstractShape> closestShape = intersection.Shape;
//         if (closestShape == null) {
//            sample.SpectralPowerDistribution = new SpectralPowerDistribution();
//            return sample;
//         }
         Normal intersectionNormal = intersection.Normal;
         Vector incomingDirection = ray.Direction;

         Vector outgoingDirection = closestShape->Material->BRDF->getVectorInPDF(intersectionNormal, incomingDirection);
         float scalePercentage = closestShape->Material->BRDF->f(incomingDirection, intersectionNormal, outgoingDirection);

         Ray bounceRay = Ray(intersection.Location, outgoingDirection);

         // fuzz fix/hack
         bounceRay.OffsetOrigin(intersectionNormal, Polytope::OffsetEpsilon);

         Sample incomingSample = GetSample(bounceRay, depth + 1, x, y);

         SpectralPowerDistribution incomingSPD = incomingSample.SpectralPowerDistribution * scalePercentage;

         // compute the interaction of the incoming SPD with the object's SRC

         SpectralPowerDistribution reflectedSPD = incomingSPD * closestShape->Material->ReflectanceSpectrum;

         return Sample(reflectedSPD);
      }
   }
}