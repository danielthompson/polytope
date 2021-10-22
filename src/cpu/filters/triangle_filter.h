//
// Created by daniel on 10/16/21.
//

#ifndef POLYTOPE_TRIANGLE_FILTER_H
#define POLYTOPE_TRIANGLE_FILTER_H


#include "abstract_filter.h"

namespace poly {

   class triangle_filter : public abstract_filter {
   public:

      explicit triangle_filter(
            const poly::bounds &bounds, 
            const float x_width,
            const float y_width)
            : abstract_filter(bounds, x_width, y_width)
            { }
      
   protected:
      float evaluate(const poly::point2i& pixel, const poly::point2f& location) const override {

         float x_dist = std::abs(location.x - ((float)pixel.x) - 0.5f);
         float y_dist = std::abs(location.y - ((float)pixel.y) - 0.5f);

         if (x_dist > x_width)
            return 0.f;

         if (y_dist > y_width)
            return 0.f;

         return ((x_width - x_dist) + (y_width - y_dist)) / (x_width + y_width);
      }
      
   };
}


#endif //POLYTOPE_TRIANGLE_FILTER_H
