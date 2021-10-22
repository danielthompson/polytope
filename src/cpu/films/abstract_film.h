//
// Created by Daniel Thompson on 3/6/18.
//

#ifndef POLY_ABSTRACTFILM_H
#define POLY_ABSTRACTFILM_H

#include <memory>
#include "../structures/Sample.h"
#include "../../common/structures/point2.h"
#include "../filters/abstract_filter.h"

namespace poly {

   class abstract_film {
   public:
      const poly::bounds Bounds;
      std::unique_ptr<poly::abstract_filter> filter;

      abstract_film(const poly::bounds bounds, std::unique_ptr<poly::abstract_filter> filter)
            : Bounds(bounds), filter(std::move(filter)) { };
      virtual ~abstract_film() = default;

      virtual void add_sample(const poly::point2i& pixel, const poly::point2f &location, const poly::Sample &sample) = 0;
      virtual void output() = 0;
   };
}

#endif //POLY_ABSTRACTFILM_H
