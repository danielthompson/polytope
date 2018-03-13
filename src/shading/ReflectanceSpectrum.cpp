//
// Created by Daniel Thompson on 2/28/18.
//

#include "ReflectanceSpectrum.h"

namespace Polytope {

   ReflectanceSpectrum &ReflectanceSpectrum::operator+=(const ReflectanceSpectrum &rhs) {
      r += rhs.r;
      g += rhs.g;
      b += rhs.b;
      return *this;
   }

   ReflectanceSpectrum ReflectanceSpectrum::operator+(const ReflectanceSpectrum &rhs) {
      return ReflectanceSpectrum(r + rhs.r, g + rhs.g, b + rhs.b);
   }

   ReflectanceSpectrum &ReflectanceSpectrum::operator*=(const ReflectanceSpectrum &rhs) {
      r *= rhs.r;
      g *= rhs.g;
      b *= rhs.b;
      return *this;
   }

   ReflectanceSpectrum ReflectanceSpectrum::operator*(const float t) {
      return ReflectanceSpectrum(r * t, g * t, b * t);
   }

   ReflectanceSpectrum ReflectanceSpectrum::operator*=(const float t) {
      r *= t;
      g *= t;
      b *= t;
      return *this;
   }

   ReflectanceSpectrum ReflectanceSpectrum::operator*(const ReflectanceSpectrum &rhs) {
      return ReflectanceSpectrum(r * rhs.r, g * rhs.g, b * rhs.b);
   }

   ReflectanceSpectrum ReflectanceSpectrum::Lerp(const float w1, const ReflectanceSpectrum &s2, const float w2) {
      return ReflectanceSpectrum(
         r * w1 + s2.r * w2,
         g * w1 + s2.g * w2,
         b * w1 + s2.b * w2);
   }

}