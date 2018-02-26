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

      float f(float thetaIncoming, float thetaOutgoing) override;
      Vector getVectorInPDF(Normal normal, Vector incoming) override;
      float f(Vector incoming, Normal normal, Vector outgoing) override;

      // data

      // TODO
      bool Delta = true;
   };

}

#endif //POLYTOPE_MIRRORBRDF_H
