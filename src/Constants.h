//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_CONSTANTS_H
#define POLYTOPE_CONSTANTS_H

#include <cmath>

namespace Polytope {

   const float Epsilon = .00001f;
   const float PI = float(M_PI);
   const float OneOverPi = 1.0f / PI;
   const float PIOver180 = PI / 180.0f;

   const float NOHIT = -1.0f;

   const float PIOver2 = PI / 2.0f;
   const float PIOver3 = PI / 3.0f;
   const float PIOver4 = PI / 4.0f;
   const float PIOver6 = PI / 6.0f;
   const float PIOver12 = PI / 12.0f;

   const float OneOver255 = 1.0f / 255.0f;

   const float Root2 = float(M_SQRT2);
   const float Root3 = sqrt(3.0f);

   bool WithinEpsilon(float number, float target, float epsilon) {

      if (number > target) {
         return (2 * epsilon + target >= number);
      }
      else
         return (2 * epsilon + number >= target);
   }

   bool WithinEpsilon(float number, float target) {

      if (number > target) {
         return (2 * Epsilon + target >= number);
      }
      else
         return (2 * Epsilon + number >= target);
   }
}

#endif //POLYTOPE_CONSTANTS_H
