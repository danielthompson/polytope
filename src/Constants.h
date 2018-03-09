//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_CONSTANTS_H
#define POLYTOPE_CONSTANTS_H

#include <cmath>

namespace Polytope {

   constexpr float Epsilon = .00001f;
   constexpr float PI = float(M_PI);
   constexpr float OneOverPi = 1.0f / PI;
   constexpr float PIOver180 = PI / 180.0f;

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


   inline float RadiansBetween(Vector &v, Normal &n) {
      //v.Normalize();
      //n.Normalize();

      return abs(acos(v.Dot(n)));
   }
}

#endif //POLYTOPE_CONSTANTS_H
