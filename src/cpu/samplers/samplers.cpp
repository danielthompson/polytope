//
// Created by Daniel Thompson on 2/21/18.
//
   
#include "samplers.h"
#include "../constants.h"

namespace poly {

   void center_sampler::get_samples(poly::point2f points[], unsigned int number, int x, int y) {
      for (int i = 0; i < number; i++) {
         points[i].x = (float)x + 0.5f;
         points[i].y = (float)y + 0.5f;
      }
   }

   void halton_sampler::get_samples(poly::point2f points[], unsigned int number, int x, int y) {
      constexpr int base0 = 2;
      constexpr int base1 = 3;

      for (int i = 1; i <= number; i++) {

         float f0 = 1;
         float f1 = 1;

         float r0 = 0.0f;
         float r1 = 0.0f;

         int index = i;

         while (index > 0) {
            f0 = f0 / base0;

            r0 = r0 + f0 * (float)(index % base0);

            index = index / base0;

         }

         index = i;

         while (index > 0) {
            f1 = f1 / base1;

            r1 = r1 + f1 * (float)(index % base1);

            index = index  / base1;
         }

         points[i - 1].x = (float)x + r0;
         points[i - 1].y = (float)y + r1;

      }
   }

   void grid_sampler::get_samples(poly::point2f points[], unsigned int number, const int x, const int y) {
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
            points[4].y = 0.7f;
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
         points[i].x += (float)x;
         points[i].y += (float)y;
      }
   }

   void random_sampler::get_samples(poly::point2f points[], unsigned int number, const int x, const int y) {
      for (int i = 0; i < number; i++) {
         points[i] = {x + normalized_uniform_random(), y + normalized_uniform_random() };
      }
   }
}