//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLY_POINTLIGHT_H
#define POLY_POINTLIGHT_H

#include "AbstractLight.h"
#include "../structures/Vectors.h"

namespace poly {

   class PointLight : public AbstractLight {
   public:
      // constructors
      explicit PointLight(const class SpectralPowerDistribution &spectralPowerDistribution, Point location)
            : AbstractLight(spectralPowerDistribution), Location(location) { }



   private:
      Point Location;
   };

}

#endif //POLY_POINTLIGHT_H
