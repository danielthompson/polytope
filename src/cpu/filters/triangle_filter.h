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
      float evaluate(const poly::point2i& pixel, const poly::point2f& location) const override;
      
   };
}


#endif //POLYTOPE_TRIANGLE_FILTER_H
