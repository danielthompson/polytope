//
// Created by Daniel Thompson on 2/21/18.
//

#include "CenterSampler.h"

namespace Polytope {

   Polytope::Point2f Polytope::CenterSampler::GetSample(int x, int y) {
      return Point2f(x + 0.5f, y + 0.5f);
   }

}