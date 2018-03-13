//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLYTOPE_MIRRORBRDF_H
#define POLYTOPE_MIRRORBRDF_H

#include "AbstractBRDF.h"

namespace Polytope {

   class MirrorBRDF : public AbstractBRDF {
   public:

      // constructors

      // operators

      // methods

      float f(float thetaIncoming, float thetaOutgoing) const override;
      Vector getVectorInPDF(const Normal &normal, const Vector &incoming) const override;
      float f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const override;

      // data

      // TODO
      bool Delta = true;
   };

}

#endif //POLYTOPE_MIRRORBRDF_H
