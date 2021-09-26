//
// Created by Daniel Thompson on 2/21/18.
//

#include "samplers.h"

namespace poly {

   point2f CenterSampler::GetSample(const int x, const int y) const {
      return point2f(x + 0.5f, y + 0.5f);
   }

   void CenterSampler::GetSamples(point2f points[], const unsigned int number, const int x, const int y) const {
      for (int i = 0; i < number; i++) {
         points[i].x = x + 0.5f;
         points[i].y = y + 0.5f;
      }
   }

   point2f HaltonSampler::GetSample(const int x, const int y) const {
      return point2f(x + 0.5f, y + 0.5f);
   }

   void HaltonSampler::GetSamples(point2f points[], const unsigned int number, const int x, const int y) const {
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

   point2f GridSampler::GetSample(const int x, const int y) const {
      return point2f(x + 0.5f, y + 0.5f);
   }

   void GridSampler::GetSamples(point2f points[], const unsigned int number, const int x, const int y) const {
      switch (number) {
         case 1: {
            points[0].x = 0.5f;
            points[0].y = 0.5f;
            break;
         }
         case 2: {
            points[0].x = 0.25f;
            points[0].y = 0.25f;

            points[1].x = 0.75f;
            points[1].y = 0.75f;
            break;
         }
         case 3: {
            if (x % 2 == y % 2) {
               points[0].x = 0.5f;
               points[0].y = 0.25f;

               points[1].x = 0.25f;
               points[1].y = 0.75f;

               points[2].x = 0.75f;
               points[2].y = 0.75f;
            }
            else {
               points[0].x = 0.25f;
               points[0].y = 0.25f;

               points[1].x = 0.75f;
               points[1].y = 0.25f;

               points[2].x = 0.5f;
               points[2].y = 0.75f;
            }

            break;
         }
         case 4: {
            points[0].x = 0.25f;
            points[0].y = 0.25f;

            points[1].x = 0.25f;
            points[1].y = 0.75f;

            points[2].x = 0.75f;
            points[2].y = 0.25f;

            points[3].x = 0.75f;
            points[3].y = 0.75f;
            break;
         }
         case 5: {
            points[0].x = 0.1f;
            points[0].y = 0.1f;

            points[1].x = 0.3f;
            points[1].y = 0.5f;

            points[2].x = 0.5f;
            points[2].y = 0.9f;

            points[3].x = 0.7f;
            points[3].y = 0.3f;

            points[4].x = 0.9f;
            points[5].y = 0.7f;
            break;
         }
         default: {
            for (int i = 0; i < number; i++) {
               points[i].x = 0.5f;
               points[i].y = 0.5f;
            }
         }
      }

      for (int i = 0; i < number; i++) {
         points[i].x += x;
         points[i].y += y;
      }
   }
}