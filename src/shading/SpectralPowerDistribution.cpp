//
// Created by Daniel Thompson on 2/28/18.
//

#include "SpectralPowerDistribution.h"
#include "../Constants.h"

namespace Polytope {

   SpectralPowerDistribution &SpectralPowerDistribution::operator+=(const SpectralPowerDistribution &rhs) {
      r += rhs.r;
      g += rhs.g;
      b += rhs.b;
      return *this;
   }

   SpectralPowerDistribution SpectralPowerDistribution::operator+(const SpectralPowerDistribution &rhs) {
      return SpectralPowerDistribution(r + rhs.r, g + rhs.g, b + rhs.b);
   }

   SpectralPowerDistribution &SpectralPowerDistribution::operator*=(const SpectralPowerDistribution &rhs) {
      r *= rhs.r;
      g *= rhs.g;
      b *= rhs.b;
      return *this;
   }

   SpectralPowerDistribution SpectralPowerDistribution::operator*(const float t) {
      return SpectralPowerDistribution(r * t, g * t, b * t);
   }

   SpectralPowerDistribution &SpectralPowerDistribution::operator*=(const float t) {
      r *= t;
      g *= t;
      b *= t;
      return *this;
   }

   SpectralPowerDistribution SpectralPowerDistribution::operator*(const SpectralPowerDistribution &rhs) {
      return SpectralPowerDistribution(r * rhs.r, g * rhs.g, b * rhs.b);
   }

   SpectralPowerDistribution SpectralPowerDistribution::operator*(const ReflectanceSpectrum &rhs) {
      return SpectralPowerDistribution(r * rhs.r * OneOver255, g * rhs.g * OneOver255, b * rhs.b * OneOver255);
   }

   SpectralPowerDistribution
   SpectralPowerDistribution::Lerp(const float w1, const SpectralPowerDistribution &s2, const float w2) {
      return SpectralPowerDistribution(
         r * w1 + s2.r * w2,
         g * w1 + s2.g * w2,
         b * w1 + s2.b * w2);
   }
}