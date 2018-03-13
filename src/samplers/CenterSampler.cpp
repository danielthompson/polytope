//
// Created by Daniel Thompson on 2/21/18.
//

#include "CenterSampler.h"

namespace Polytope {

   Point2f CenterSampler::GetSample(const int x, const int y) {
      return Point2f(x + 0.5f, y + 0.5f);
   }


}