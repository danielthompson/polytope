//
// Created by Daniel Thompson on 2/24/18.
//

#include "MirrorBRDF.h"
#include "../../Constants.h"

namespace Polytope {

   float MirrorBRDF::f(const float thetaIncoming, const float thetaOutgoing) const {
      if (Polytope::WithinEpsilon(thetaIncoming, thetaOutgoing))
         return 1.0f;
      return 0.0f;
   }

   Vector MirrorBRDF::getVectorInPDF(const Normal &normal, const Vector &incoming, float &pdf) const {
      float factor = incoming.Dot(normal) * 2;
      Vector scaled = Vector(normal * factor);

      pdf = 1.0f;
      Vector outgoing = incoming - scaled;
      return outgoing;

   }

   float MirrorBRDF::f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const {

      float thetaIncoming = RadiansBetween(-incoming, normal);
      float thetaOutgoing = RadiansBetween(outgoing, normal);

      return f(thetaIncoming, thetaOutgoing);
   }

}