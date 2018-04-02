//
// Created by Daniel on 31-Mar-18.
//

#include "AbstractBRDF.h"
#include "../../Constants.h"

namespace Polytope {

   Vector AbstractBRDF::getVectorInPDF(const Normal &normal, const Vector &incoming, float &pdf) const {
      const float u0 = NormalizedUniformRandom();
      const float u1 = NormalizedUniformRandom();

      const Vector hemi = CosineSampleHemisphere(u0, u1);

      pdf = OneOverPi;

      return hemi;
   }
}