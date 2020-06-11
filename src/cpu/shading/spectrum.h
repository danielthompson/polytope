//
// Created by daniel on 5/7/20.
//

#ifndef POLY_SPECTRUM_H
#define POLY_SPECTRUM_H

#include "../constants.h"

namespace poly {
   class ReflectanceSpectrum {
   public:

      // constructors
      explicit ReflectanceSpectrum() : r(1.f), g(1.f), b(1.f) { };
      ReflectanceSpectrum(float r, float g, float b) : r(r), g(g), b(b) { };
      ReflectanceSpectrum(int r, int g, int b)
            : r(r * poly::OneOver255), g(g * poly::OneOver255), b(b * poly::OneOver255) { };

      bool is_black() {
         return (r <= 0.0001f && g <= 0.0001f && b <= 0.0001f);
      }
      
      ReflectanceSpectrum& operator+=(const ReflectanceSpectrum &rhs) {
         r += rhs.r;
         g += rhs.g;
         b += rhs.b;
         return *this;
      }

      ReflectanceSpectrum operator+(const ReflectanceSpectrum &rhs) const {
         return {r + rhs.r, g + rhs.g, b + rhs.b};
      }

      ReflectanceSpectrum& operator*=(const ReflectanceSpectrum &rhs) {
         r *= rhs.r;
         g *= rhs.g;
         b *= rhs.b;
         return *this;
      }

      ReflectanceSpectrum operator*(const float t) const {
         return {r * t, g * t, b * t};
      }

      ReflectanceSpectrum& operator*=(const float t) {
         r *= t;
         g *= t;
         b *= t;
         return *this;
      }

      ReflectanceSpectrum operator*(const ReflectanceSpectrum &rhs) const {
         return {r * rhs.r, g * rhs.g, b * rhs.b};
      }

      ReflectanceSpectrum Lerp(const float w1, const ReflectanceSpectrum &s2, const float w2) const {
         return {
               r * w1 + s2.r * w2,
               g * w1 + s2.g * w2,
               b * w1 + s2.b * w2
         };
      }

      // data
      float r, g, b;
   };

   class SpectralPowerDistribution {
   public:

      // constructors
      explicit SpectralPowerDistribution() : r(0), g(0), b(0) { };
      SpectralPowerDistribution(float r, float g, float b) : r(r), g(g), b(b) { };
      SpectralPowerDistribution(const SpectralPowerDistribution &other) = default;

      SpectralPowerDistribution& operator+=(const SpectralPowerDistribution &rhs) {
         r += rhs.r;
         g += rhs.g;
         b += rhs.b;
         return *this;
      }

      SpectralPowerDistribution operator+(const SpectralPowerDistribution &rhs) const {
         return SpectralPowerDistribution(r + rhs.r, g + rhs.g, b + rhs.b);
      }

      SpectralPowerDistribution& operator*=(const SpectralPowerDistribution &rhs) {
         r *= rhs.r;
         g *= rhs.g;
         b *= rhs.b;
         return *this;
      }

      SpectralPowerDistribution operator*(const float t) const {
         return SpectralPowerDistribution(r * t, g * t, b * t);
      }

      SpectralPowerDistribution& operator*=(const float t) {
         r *= t;
         g *= t;
         b *= t;
         return *this;
      }

      SpectralPowerDistribution operator*(const SpectralPowerDistribution &rhs) {
         return SpectralPowerDistribution(r * rhs.r, g * rhs.g, b * rhs.b);
      }

      SpectralPowerDistribution operator*(const ReflectanceSpectrum &rhs) {
         return SpectralPowerDistribution(r * rhs.r, g * rhs.g, b * rhs.b);
      }

      SpectralPowerDistribution Lerp(const float w1, const SpectralPowerDistribution &s2, const float w2) {
         return SpectralPowerDistribution(
               r * w1 + s2.r * w2,
               g * w1 + s2.g * w2,
               b * w1 + s2.b * w2);
      }

      // data
      float r, g, b;
   };
}

#endif //POLY_SPECTRUM_H
