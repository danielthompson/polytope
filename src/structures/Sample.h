//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLYTOPE_SAMPLE_H
#define POLYTOPE_SAMPLE_H

#include "../shading/SpectralPowerDistribution.h"

namespace Polytope {

   class Sample {
   public:

      // constructors
      Sample() : SpectralPowerDistribution() { }

      explicit Sample(const SpectralPowerDistribution &spectralPowerDistribution)
            : SpectralPowerDistribution(spectralPowerDistribution) { }

      // data
      SpectralPowerDistribution SpectralPowerDistribution;
   };

}


#endif //POLYTOPE_SAMPLE_H
