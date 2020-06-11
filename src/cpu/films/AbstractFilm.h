//
// Created by Daniel Thompson on 3/6/18.
//

#ifndef POLY_ABSTRACTFILM_H
#define POLY_ABSTRACTFILM_H

#include <memory>
#include "../structures/Sample.h"
#include "../../common/structures/Point2.h"
#include "../filters/AbstractFilter.h"

namespace poly {

   class AbstractFilm {
   public:
      const poly::Bounds Bounds;
      std::unique_ptr<AbstractFilter> Filter;

      AbstractFilm(const poly::Bounds bounds, std::unique_ptr<AbstractFilter> filter)
            : Bounds(bounds), Filter(std::move(filter)) { };
      virtual ~AbstractFilm() = default;

      virtual void AddSample(const Point2f &location, const Sample &sample) = 0;
      virtual void Output() = 0;
   };
}

#endif //POLY_ABSTRACTFILM_H
