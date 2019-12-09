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

      virtual float f(float thetaIncoming, float thetaOutgoing) const = 0;

      /**
       * Gets an outgoing vector according to the BxDF's PDF.
       * Should be used for both delta and non-delta.
       * @param incoming The incoming vector.
       * @param pdf The pdf of the returned vector.
       * @return A vector randomly samplb
       */
      virtual Vector getVectorInPDF(Vector incoming, float &pdf) const;
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
      virtual float f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const = 0;

      virtual ~AbstractBRDF() = default;

   };

}

#endif //POLYTOPE_ABSTRACTBRDF_H
