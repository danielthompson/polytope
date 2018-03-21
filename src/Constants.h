//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_CONSTANTS_H
#define POLYTOPE_CONSTANTS_H

#include <cmath>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

namespace Polytope {

   constexpr float Epsilon = .00001f;
   constexpr float HalfEpsilon = Epsilon * 0.5f;
   constexpr float OffsetEpsilon = 0.002f;
   constexpr float PI = float(M_PI);
   constexpr float OneOverPi = 1.0f / PI;
   constexpr float PIOver180 = PI / 180.0f;

   constexpr float OneThird = 1.0f / 3.0f;

   constexpr float NOHIT = -1.0f;

   constexpr float PIOver2 = PI / 2.0f;
   constexpr float PIOver3 = PI / 3.0f;
   constexpr float PIOver4 = PI / 4.0f;
   constexpr float PIOver6 = PI / 6.0f;
   constexpr float PIOver12 = PI / 12.0f;

   constexpr float OneOver255 = 1.0f / 255.0f;

   constexpr float Root2 = float(M_SQRT2);

   constexpr float Infinity = std::numeric_limits<float>::infinity();
   constexpr float DenormMin = std::numeric_limits<float>::denorm_min();

   const float Root3 = sqrt(3.0f);

   inline bool WithinEpsilon(float number, float target, float epsilon) {

      if (number > target) {
         return (2 * epsilon + target >= number);
      }
      else
         return (2 * epsilon + number >= target);
   }

   inline bool WithinEpsilon(float number, float target) {

      if (number > target) {
         return (2 * Epsilon + target >= number);
      }
      else
         return (2 * Epsilon + number >= target);
   }


   inline float RadiansBetween(const Vector &v, const Normal &n) {
      //v.Normalize();
      //n.Normalize();

      return float(std::abs(acos(v.Dot(n))));
   }
}

#endif //POLYTOPE_CONSTANTS_H
