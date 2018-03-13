//
// Created by Daniel Thompson on 2/28/18.
//

#ifndef POLYTOPE_SPECTRALPOWERDISTRIBUTION_H
#define POLYTOPE_SPECTRALPOWERDISTRIBUTION_H

#include "../shading/ReflectanceSpectrum.h"

namespace Polytope {

   class SpectralPowerDistribution {
   public:

      // constructors
      explicit SpectralPowerDistribution() : r(0), g(0), b(0) { };
      SpectralPowerDistribution(float r, float g, float b) : r(r), g(g), b(b) { };
      SpectralPowerDistribution(const SpectralPowerDistribution &other) : r(other.r), g(other.g), b(other.b) { }

      // operators
      SpectralPowerDistribution &operator+=(const SpectralPowerDistribution &rhs);
      SpectralPowerDistribution operator+(const SpectralPowerDistribution &rhs);
      SpectralPowerDistribution &operator*=(const SpectralPowerDistribution &rhs);
      SpectralPowerDistribution operator*(const float t);
      SpectralPowerDistribution &operator*=(const float t);
      SpectralPowerDistribution operator*(const SpectralPowerDistribution &rhs);
      SpectralPowerDistribution operator*(const ReflectanceSpectrum &rhs);

      // methods
      SpectralPowerDistribution Lerp(const float w1, const SpectralPowerDistribution &s2, const float w2);

      // data
      float r, g, b;
   };

}

#endif //POLYTOPE_SPECTRALPOWERDISTRIBUTION_H
