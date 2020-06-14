//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLY_ABSTRACTBRDF_H
#define POLY_ABSTRACTBRDF_H

#include "../../constants.h"
#include "../../structures/Vectors.h"
#include "../spectrum.h"

namespace poly {

   class AbstractBRDF {
   public:
      virtual float f(float thetaIncoming, float thetaOutgoing) const = 0;

      /**
       * Gets an outgoing vector according to the BxDF's PDF.
       * Should be used for both delta and non-delta.
       * @param incoming The incoming vector.
       * @param pdf The pdf of the returned vector.
       * @return A vector randomly samplb
       */
      virtual Vector sample(const Vector &incoming, poly::ReflectanceSpectrum &refl_spectrum, float &pdf) const {
         const float u0 = NormalizedUniformRandom();
         const float u1 = NormalizedUniformRandom();

         const Vector hemi = CosineSampleHemisphere(u0, u1);

         pdf = 1;
         
         return hemi;
      }
      
      /**
       * Returns the proportion of outgoing light that comes from the incoming direction.
       * Should only be used for non-delta distributions - f() can be assumed to be 0 for deltas
       * except when the outgoing Vector has been obtained from sample(), in which case
       * there should be no need to call this since it can be assumed to be 1.
       * @param incoming The direction of incoming light.
       * @param normal The surface normal at the point of intersection.
       * @param outgoing The direction of outgoing light.
       * @return The proportion of outgoing light that comes from the incoming direction.
       */
      virtual float f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const = 0;

      virtual ~AbstractBRDF() = default;

   };

}

#endif //POLY_ABSTRACTBRDF_H
