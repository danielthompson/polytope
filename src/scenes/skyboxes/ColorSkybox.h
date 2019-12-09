//
// Created by Daniel Thompson on 2019-06-26.
//

#ifndef POLYTOPE_COLORSKYBOX_H
#define POLYTOPE_COLORSKYBOX_H

#include "AbstractSkybox.h"

class ColorSkybox : AbstractSkybox {
public:
   explicit ColorSkybox(const Polytope::SpectralPowerDistribution& spd) : spd(spd) {  }

   Polytope::SpectralPowerDistribution GetSpd(const Polytope::Vector &v) const;

private:
   Polytope::SpectralPowerDistribution spd;
};


#endif //POLYTOPE_COLORSKYBOX_H
