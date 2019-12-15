//
// Created by Daniel on 04-Apr-18.
//

#include "HaltonSampler.h"

namespace Polytope {

   Point2f HaltonSampler::GetSample(const int x, const int y) const {
      return Point2f(x + 0.5f, y + 0.5f);
   }

   void HaltonSampler::GetSamples(Point2f points[], const unsigned int number, const int x, const int y) const {
      const int base0 = 2;
      const int base1 = 3;

      for (int i = 1; i <= number; i++) {

         float f0 = 1;
         float f1 = 1;

         float r0 = 0.0f;
         float r1 = 0.0f;

         int index = i;

         while (index > 0) {
            f0 = f0 / base0;

            r0 = r0 + f0 * (index % base0);

            index = index / base0;

         }

         index = i;

         while (index > 0) {
            f1 = f1 / base1;

            r1 = r1 + f1 * (index % base1);

            index = index  / base1;
         }

         points[i - 1].x = x + r0;
         points[i - 1].y = y + r1;

      }
   }
}