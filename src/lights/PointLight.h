//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLYTOPE_POINTLIGHT_H
#define POLYTOPE_POINTLIGHT_H

#include "AbstractLight.h"

namespace Polytope {

   class PointLight : public AbstractLight {
   public:
      // constructors
      explicit PointLight(const class SpectralPowerDistribution &spectralPowerDistribution, Point location)
            : AbstractLight(spectralPowerDistribution), Location(location) { }

      Point GetRandomPointOnSurface() override;

   private:
      Point Location;
   };

}

#endif //POLYTOPE_POINTLIGHT_H
