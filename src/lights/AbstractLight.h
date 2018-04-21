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
      explicit AbstractLight(const Polytope::SpectralPowerDistribution &spectralPowerDistribution) :
            SpectralPowerDistribution(spectralPowerDistribution) {  };


      // data
      Polytope::SpectralPowerDistribution SpectralPowerDistribution;
   };

}


#endif //POLYTOPE_ABSTRACTLIGHT_H
