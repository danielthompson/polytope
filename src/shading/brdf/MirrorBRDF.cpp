//
// Created by Daniel Thompson on 2/24/18.
//

#include "MirrorBRDF.h"
#include "../../Constants.h"

namespace Polytope {

   float MirrorBRDF::f(float thetaIncoming, float thetaOutgoing) {
      if (Polytope::WithinEpsilon(thetaIncoming, thetaOutgoing))
         return 1.0f;
      return 0.0f;
   }

   Vector MirrorBRDF::getVectorInPDF(Normal normal, Vector incoming) {
      normal.Normalize();
      float factor = incoming.Dot(normal) * 2;
      Vector scaled = Vector(normal * factor);

      //Vector outgoing = Vector.Minus(incoming, scaled);

      // TODO combine these two statements
      Vector outgoing = scaled - incoming;
      outgoing = -outgoing;

      return outgoing;
   }

   float MirrorBRDF::f(Vector incoming, Normal normal, Vector outgoing) {

      // Vector fixedIncoming = Vector.Scale(incoming, -1);
      incoming = -incoming;

      float thetaIncoming = RadiansBetween(incoming, normal);
      float thetaOutgoing = RadiansBetween(outgoing, normal);

      return f(thetaIncoming, thetaOutgoing);
   }

}