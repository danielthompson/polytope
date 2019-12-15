//
// Created by Daniel Thompson on 3/5/18.
//

#include <iostream>
#include "PathTraceIntegrator.h"

namespace Polytope {

   Sample PathTraceIntegrator::GetSample(Ray &ray, int depth, int x, int y) {

      Intersection intersection = Scene->GetNearestShape(ray, x, y);

      if (!intersection.Hits) {

         SpectralPowerDistribution spd;

         if (Scene->Skybox != nullptr) {
            spd = Scene->Skybox->GetSpd(ray.Direction);
         }

         return Sample(spd);
      }

      if (intersection.Shape->IsLight()) {

         return Sample(intersection.Shape->Light->SpectralPowerDistribution);
      }

      // base case
      if (depth >= MaxDepth) {
         return Sample(SpectralPowerDistribution());
      } else {

         // indirect lighting

         float pdf = 0.0f;
         Vector outgoingDirection = intersection.LocalToWorld(
               intersection.Shape->Material->BRDF->getVectorInPDF(
                  intersection.WorldToLocal(ray.Direction), pdf
               )
            );

         Ray bounceRay = Ray(intersection.Location, outgoingDirection);

         // fuzz fix/hack
         bounceRay.OffsetOrigin(intersection.Normal, Polytope::OffsetEpsilon);

         Sample incomingSample = GetSample(bounceRay, depth + 1, x, y);

         SpectralPowerDistribution incomingSPD = incomingSample.SpectralPowerDistribution * pdf;

         // compute the interaction of the incoming SPD with the object's SRC
         SpectralPowerDistribution reflectedSPD = incomingSPD * intersection.Shape->Material->ReflectanceSpectrum;


         return Sample(reflectedSPD);
      }
   }
}