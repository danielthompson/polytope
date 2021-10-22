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
      std::unique_ptr<abstract_filter> Filter;

      abstract_film(const poly::bounds bounds, std::unique_ptr<abstract_filter> filter)
            : Bounds(bounds), Filter(std::move(filter)) { };
      virtual ~abstract_film() = default;

      virtual void AddSample(const point2i& pixel, const point2f &location, const Sample &sample) = 0;
      virtual void Output() = 0;
   };
}

#endif //POLY_ABSTRACTFILM_H