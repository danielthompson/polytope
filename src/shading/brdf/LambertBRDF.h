//
// Created by Daniel on 31-Mar-18.
//

#ifndef POLYTOPE_LAMBERTBRDF_H
#define POLYTOPE_LAMBERTBRDF_H

#include "AbstractBRDF.h"
#include "../../Constants.h"

namespace Polytope {

   class LambertBRDF : public AbstractBRDF {
   public:
      float f(float thetaIncoming, float thetaOutgoing) const override;
      float f(const Vector &incoming, const Normal &normal, const Vector &outgoing) const override;

   };

}


#endif //POLYTOPE_LAMBERTBRDF_H
