//
// Created by Daniel on 16-Mar-18.
//

#include "GridSampler.h"

namespace Polytope {


   Point2f GridSampler::GetSample(int x, int y) {
      return Point2f(x + 0.5f, y + 0.5f);
   }

   void GridSampler::GetSamples(Point2f points[], int number, int x, int y) {
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