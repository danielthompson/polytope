//
// Created by Daniel Thompson on 2/28/18.
//

#ifndef POLYTOPE_REFLECTANCESPECTRUM_H
#define POLYTOPE_REFLECTANCESPECTRUM_H

namespace Polytope {

   class ReflectanceSpectrum {
   public:
      // constructors

      explicit ReflectanceSpectrum() : r(0), g(0), b(0) { };
      ReflectanceSpectrum(float r, float g, float b) : r(r), g(g), b(b) { };

      // operators

      ReflectanceSpectrum &operator+=(const ReflectanceSpectrum &rhs) {
         r += rhs.r;
         g += rhs.g;
         b += rhs.b;
         return *this;
      }

      ReflectanceSpectrum operator+(const ReflectanceSpectrum &rhs) {
         return ReflectanceSpectrum(r + rhs.r, g + rhs.g, b + rhs.b);
      }

      ReflectanceSpectrum &operator*=(const ReflectanceSpectrum &rhs) {
         r *= rhs.r;
         g *= rhs.g;
         b *= rhs.b;
         return *this;
      }

      ReflectanceSpectrum operator*(const float t) {
         return ReflectanceSpectrum(r * t, g * t, b * t);
      }

      ReflectanceSpectrum operator*=(const float t) {
         r *= t;
         g *= t;
         b *= t;
         return *this;
      }

      ReflectanceSpectrum operator*(const ReflectanceSpectrum &rhs) {
         return ReflectanceSpectrum(r * rhs.r, g + rhs.g, b + rhs.b);
      }

      // methods

      ReflectanceSpectrum Lerp(const float w1, const ReflectanceSpectrum &s2, const float w2) {
         return ReflectanceSpectrum(
               r * w1 + s2.r * w2,
               g * w1 + s2.g * w2,
               b * w1 + s2.b * w2);
      }

      float r, g, b;
   };

}


#endif //POLYTOPE_REFLECTANCESPECTRUM_H
