//
// Created by Daniel on 31-Mar-18.
//

#include "LambertBRDF.h"
#include "../../Constants.h"

namespace Polytope {

   float LambertBRDF::f(float thetaIncoming, float thetaOutgoing) const {
      return Polytope::OneOverPi;
   }

   float LambertBRDF::f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const {
      return Polytope::OneOverPi;
   }
}