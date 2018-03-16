//
// Created by Daniel Thompson on 2/21/18.
//

#include "CenterSampler.h"

namespace Polytope {

   Point2f CenterSampler::GetSample(const int x, const int y) {
      return Point2f(x + 0.5f, y + 0.5f);
   }

   void CenterSampler::GetSamples(Point2f *points, int number, int x, int y) {
      for (int i = 0; i < number; i++) {
         points->x = x + 0.5f;
         points->y = y + 0.5f;
      }
   }


}