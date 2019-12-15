//
// Created by Daniel Thompson on 2/21/18.
//

#include "CenterSampler.h"

namespace Polytope {

   Point2f CenterSampler::GetSample(const int x, const int y) const {
      return Point2f(x + 0.5f, y + 0.5f);
   }

   void CenterSampler::GetSamples(Point2f points[], const unsigned int number, const int x, const int y) const {
      for (int i = 0; i < number; i++) {
         points[i].x = x + 0.5f;
         points[i].y = y + 0.5f;
      }
   }


}