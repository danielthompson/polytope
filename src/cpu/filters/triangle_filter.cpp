//
// Created by daniel on 10/16/21.
//

#include "triangle_filter.h"

namespace poly {

   float triangle_filter::evaluate(const poly::point2i& pixel, const poly::point2f& location) const {
      
      float x_dist = std::abs(location.x - ((float)pixel.x) - 0.5f);
      float y_dist = std::abs(location.y - ((float)pixel.y) - 0.5f);
      
      if (x_dist > x_width)
         return 0.f;
      
      if (y_dist > y_width)
         return 0.f;
      
      return ((x_width - x_dist) + (y_width - y_dist)) / (x_width + y_width); 
   }
}
