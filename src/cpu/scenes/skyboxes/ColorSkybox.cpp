//
// Created by Daniel Thompson on 2019-06-26.
//

#include "ColorSkybox.h"

namespace Polytope {
   Polytope::SpectralPowerDistribution ColorSkybox::GetSpd(const Polytope::Vector &v) const {
      return spd;
   }
}
