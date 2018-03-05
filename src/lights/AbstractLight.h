//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLYTOPE_ABSTRACTLIGHT_H
#define POLYTOPE_ABSTRACTLIGHT_H

#include "../shading/SpectralPowerDistribution.h"
#include "../structures/Point.h"

namespace Polytope {

   class AbstractLight {
   public:

      // constructors
      explicit AbstractLight(const SpectralPowerDistribution &spectralPowerDistribution) :
            SpectralPowerDistribution(spectralPowerDistribution) {  };

      // methods
      virtual Point GetRandomPointOnSurface() = 0;

      // data
      SpectralPowerDistribution SpectralPowerDistribution;
   };

}


#endif //POLYTOPE_ABSTRACTLIGHT_H
