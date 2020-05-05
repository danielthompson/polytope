//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLYTOPE_SAMPLE_H
#define POLYTOPE_SAMPLE_H

#include <memory>
#include "../shading/SpectralPowerDistribution.h"

namespace Polytope {

   class Sample {
   public:

      // constructors
      Sample() : SpectralPowerDistribution() { }

      Sample(const Polytope::SpectralPowerDistribution &spectralPowerDistribution)
            : SpectralPowerDistribution(spectralPowerDistribution) { }

      Sample(const std::shared_ptr<Polytope::SpectralPowerDistribution> spd) {
         if (spd == nullptr) {
            SpectralPowerDistribution = Polytope::SpectralPowerDistribution();
         }
         else {
            SpectralPowerDistribution = (*spd);
         }
      }
      Polytope::SpectralPowerDistribution SpectralPowerDistribution;
   };
}

#endif //POLYTOPE_SAMPLE_H
