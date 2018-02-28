//
// Created by Daniel Thompson on 2/28/18.
//

#ifndef POLYTOPE_SPECTRALPOWERDISTRIBUTION_H
#define POLYTOPE_SPECTRALPOWERDISTRIBUTION_H

namespace Polytope {

   class SpectralPowerDistribution {
   public:
      // constructors

      explicit SpectralPowerDistribution() : r(0), g(0), b(0) { };
      SpectralPowerDistribution(float r, float g, float b) : r(r), g(g), b(b) { };

      // operators

      SpectralPowerDistribution &operator+=(const SpectralPowerDistribution &rhs) {
         r += rhs.r;
         g += rhs.g;
         b += rhs.b;
         return *this;
      }

      SpectralPowerDistribution operator+(const SpectralPowerDistribution &rhs) {
         return SpectralPowerDistribution(r + rhs.r, g + rhs.g, b + rhs.b);
      }

      SpectralPowerDistribution &operator*=(const SpectralPowerDistribution &rhs) {
         r *= rhs.r;
         g *= rhs.g;
         b *= rhs.b;
         return *this;
      }

      SpectralPowerDistribution operator*(const float t) {
         return SpectralPowerDistribution(r * t, g * t, b * t);
      }

      SpectralPowerDistribution operator*=(const float t) {
         r *= t;
         g *= t;
         b *= t;
         return *this;
      }

      SpectralPowerDistribution operator*(const SpectralPowerDistribution &rhs) {
         return SpectralPowerDistribution(r * rhs.r, g + rhs.g, b + rhs.b);
      }

      // methods

      SpectralPowerDistribution Lerp(const float w1, const SpectralPowerDistribution &s2, const float w2) {
         return SpectralPowerDistribution(
               r * w1 + s2.r * w2,
               g * w1 + s2.g * w2,
               b * w1 + s2.b * w2);
      }

      float r, g, b;
   };

}

#endif //POLYTOPE_SPECTRALPOWERDISTRIBUTION_H
