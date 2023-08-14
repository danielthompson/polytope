//
// Created by Daniel Thompson on 2019-06-26.
//

#ifndef POLY_ABSTRACTSKYBOX_H
#define POLY_ABSTRACTSKYBOX_H

#include "../../shading/spectrum.h"

namespace poly {
   class AbstractSkybox {

   public:
      virtual poly::SpectralPowerDistribution GetSpd(const poly::vector &v) const = 0;

      virtual ~AbstractSkybox() { }
   };
}

#endif //POLY_ABSTRACTSKYBOX_H
