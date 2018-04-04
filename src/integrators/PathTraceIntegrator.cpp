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
      } else {
         AbstractShape *closestShape = intersection.Shape;

         Normal intersectionNormal = intersection.Normal;
         Vector incomingDirection = ray.Direction;

         float pdf = 0.0f;
         Vector outgoingDirection = intersection.LocalToWorld(
               closestShape->Material->BRDF->getVectorInPDF(
                  intersection.WorldToLocal(incomingDirection), pdf
               )
            );

         Ray bounceRay = Ray(intersection.Location, outgoingDirection);

         // fuzz fix/hack
         bounceRay.OffsetOrigin(intersectionNormal, Polytope::OffsetEpsilon);

         Sample incomingSample = GetSample(bounceRay, depth + 1, x, y);

         SpectralPowerDistribution incomingSPD = incomingSample.SpectralPowerDistribution * pdf;

         // compute the interaction of the incoming SPD with the object's SRC
         SpectralPowerDistribution reflectedSPD = incomingSPD * closestShape->Material->ReflectanceSpectrum;

         return Sample(reflectedSPD);
      }
   }
}