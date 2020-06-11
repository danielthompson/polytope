//
// Created by Daniel Thompson on 2019-06-26.
//

#ifndef POLY_COLORSKYBOX_H
#define POLY_COLORSKYBOX_H

#include "AbstractSkybox.h"

namespace poly {
   class ColorSkybox : public AbstractSkybox {
   public:
      explicit ColorSkybox(const poly::SpectralPowerDistribution &spd) : spd(spd) {}

      poly::SpectralPowerDistribution GetSpd(const poly::Vector &v) const override;

   private:
      poly::SpectralPowerDistribution spd;
   };
}

#endif //POLY_COLORSKYBOX_H
