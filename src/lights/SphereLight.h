//
// Created by Daniel Thompson on 3/10/18.
//

#ifndef POLYTOPE_SPHERELIGHT_H
#define POLYTOPE_SPHERELIGHT_H

#include "AbstractLight.h"
#include "../shapes/Sphere.h"

namespace Polytope {

   class SphereLight : public AbstractLight {
   public:
      SphereLight(const Polytope::SpectralPowerDistribution &spectralPowerDistribution,
                  const Polytope::Sphere *sphere)
         : AbstractLight(spectralPowerDistribution), Sphere(sphere) {  };

      Point GetRandomPointOnSurface() const override;

      const Polytope::Sphere *Sphere;

   };

}


#endif //POLYTOPE_SPHERELIGHT_H
