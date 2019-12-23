//
// Created by Daniel Thompson on 2019-06-26.
//

#ifndef POLYTOPE_ABSTRACTSKYBOX_H
#define POLYTOPE_ABSTRACTSKYBOX_H

#include "../../shading/SpectralPowerDistribution.h"

namespace Polytope {
   class AbstractSkybox {

   public:
      virtual Polytope::SpectralPowerDistribution GetSpd(const Polytope::Vector &v) const = 0;

      virtual ~AbstractSkybox() { }
   };
}

#endif //POLYTOPE_ABSTRACTSKYBOX_H
