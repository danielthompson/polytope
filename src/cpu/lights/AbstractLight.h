//
// Created by Daniel Thompson on 3/5/18.
//

#ifndef POLY_ABSTRACTLIGHT_H
#define POLY_ABSTRACTLIGHT_H

#include "../shading/spectrum.h"
#include "../structures/Vectors.h"

namespace poly {

   class AbstractLight {
   public:

      // constructors
      explicit AbstractLight(const poly::SpectralPowerDistribution &spectralPowerDistribution) :
            SpectralPowerDistribution(spectralPowerDistribution) {  };


      // data
      poly::SpectralPowerDistribution SpectralPowerDistribution;
   };

}


#endif //POLY_ABSTRACTLIGHT_H
