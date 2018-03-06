//
// Created by Daniel Thompson on 3/5/18.
//

#include "PathTraceIntegrator.h"

namespace Polytope {

   Sample PathTraceIntegrator::GetSample(Ray ray, int depth, int x, int y) {
      Intersection closestStateToRay = Scene->GetNearestShape(ray, x, y);

      if (!closestStateToRay.Hits) {
         return Sample(SpectralPowerDistribution());
      }

      // TODO
//      if (closestStateToRay.Shape instanceof AbstractLight) {
//         sample.SpectralPowerDistribution = ((AbstractLight) closestStateToRay.Shape).SpectralPowerDistribution;
//         return sample;
//      }

      // base case
      if (depth >= MaxDepth) {
         return Sample(SpectralPowerDistribution());
      }
      else {
         std::shared_ptr<AbstractShape> closestShape = closestStateToRay.Shape;
//         if (closestShape == null) {
//            sample.SpectralPowerDistribution = new SpectralPowerDistribution();
//            return sample;
//         }
         Normal intersectionNormal = closestStateToRay.Normal;
         Vector incomingDirection = ray.Direction;
//
//         if (x == 476 && y == 531) {
//            int j = 0;
//            Vector outgoingDirection = objectMaterial.BRDF.getVectorInPDF(intersectionNormal, incomingDirection);
//         }

         Vector outgoingDirection = closestShape->Material->BRDF->getVectorInPDF(intersectionNormal, incomingDirection);
         float scalePercentage = closestShape->Material->BRDF->f(incomingDirection, intersectionNormal, outgoingDirection);

         Ray bounceRay = Ray(closestStateToRay.Location, outgoingDirection);

         // TODO for fuzz
         // bounceRay.OffsetOriginForward(Constants.HalfEpsilon);

         Sample incomingSample = GetSample(bounceRay, depth + 1, x, y);

         SpectralPowerDistribution incomingSPD = incomingSample.SpectralPowerDistribution * scalePercentage;

         // compute the interaction of the incoming SPD with the object's SRC

         SpectralPowerDistribution reflectedSPD = incomingSPD * closestShape->Material->ReflectanceSpectrum;

         return Sample(reflectedSPD);
      }
   }
}