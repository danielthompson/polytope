//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLY_SAMPLE_H
#define POLY_SAMPLE_H

#include <memory>
#include "../shading/spectrum.h"

namespace poly {

   class Sample {
   public:

      // constructors
      Sample() : SpectralPowerDistribution() { }

      Sample(const poly::SpectralPowerDistribution &spectralPowerDistribution)
            : SpectralPowerDistribution(spectralPowerDistribution) { }

      Sample(const std::shared_ptr<poly::SpectralPowerDistribution> spd) {
         if (spd == nullptr) {
            SpectralPowerDistribution = poly::SpectralPowerDistribution();
         }
         else {
            SpectralPowerDistribution = (*spd);
         }
      }
      poly::SpectralPowerDistribution SpectralPowerDistribution;
   };
}

#endif //POLY_SAMPLE_H
