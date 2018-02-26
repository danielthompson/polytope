//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLYTOPE_ABSTRACTBRDF_H
#define POLYTOPE_ABSTRACTBRDF_H

#include "../../structures/Vector.h"
#include "../../structures/Normal.h"

namespace Polytope {

   class AbstractBRDF {
   public:

      // constructors

      // operators

      // methods

      virtual float f(float thetaIncoming, float thetaOutgoing) = 0;

      /**
       * Gets an outgoing vector according to the BxDF's PDF.
       * Should be used for both delta and non-delta.
       * @param normal The surface normal at the point of intersection.
       * @param incoming The incoming vector.
       * @return A vector randomly sampled according to the BxDF's PDF.
       */
      virtual Vector getVectorInPDF(Normal normal, Vector incoming) = 0;

      /**
       * Returns the proportion of outgoing light that comes from the incoming direction.
       * Should only be used for non-delta distributions - f() can be assumed to be 0 for deltas
       * except when the outgoing Vector has been obtained from getVectorInPDF(), in which case
       * there should be no need to call this since it can be assumed to be 1.
       * @param incoming The direction of incoming light.
       * @param normal The surface normal at the point of intersection.
       * @param outgoing The direction of outgoing light.
       * @return The proportion of outgoing light that comes from the incoming direction.
       */
      virtual float f(Vector incoming, Normal normal, Vector outgoing) = 0;


      // data

      /**
       * Whether or not the distribution is a delta distribution.
       * If so, it should be sampled with f_delta(), not f().
       */
      bool Delta = false;

      bool Glossy = false;

   };

}

#endif //POLYTOPE_ABSTRACTBRDF_H
