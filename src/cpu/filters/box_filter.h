//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLY_BOXFILTER_H
#define POLY_BOXFILTER_H

#include <vector>
#include <mutex>
#include "abstract_filter.h"
#include "../../common/structures/point2.h"

namespace poly {

   class box_filter : public abstract_filter {
   public:

      explicit box_filter(
            const poly::bounds &bounds,
            const float x_width,
            const float y_width)
            : abstract_filter(bounds, x_width, y_width) { }

   protected:
      float evaluate(const point2i& pixel, const point2f& location) const override;
   };
}

#endif //POLY_BOXFILTER_H
