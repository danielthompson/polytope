//
// Created by Daniel on 16-Mar-18.
//

#include "BoxFilter.h"
#include "../structures/Point2.h"

namespace Polytope {

   void BoxFilter::AddSample(const Point2f &location, const Sample &sample) {
      const int x = static_cast<int>(location.x);
      const int y = static_cast<int>(location.y);

      _data[x][y][0] = sample;
   }

   Sample BoxFilter::Output(const Point2i &pixel) {
      return _data[pixel.x][pixel.y][0];
   }
}