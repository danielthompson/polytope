//
// Created by Daniel Thompson on 2019-06-26.
//

#include "ColorSkybox.h"

namespace poly {
   poly::SpectralPowerDistribution ColorSkybox::GetSpd(const poly::vector &v) const {
      return spd;
   }
}
