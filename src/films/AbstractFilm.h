//
// Created by Daniel Thompson on 3/6/18.
//

#ifndef POLYTOPE_ABSTRACTFILM_H
#define POLYTOPE_ABSTRACTFILM_H

#include <memory>
#include "../structures/Sample.h"
#include "../structures/Point2.h"
#include "../filters/AbstractFilter.h"

namespace Polytope {

   class AbstractFilm {
   public:

      // constructors
      AbstractFilm(const Polytope::Bounds bounds, std::unique_ptr<AbstractFilter> filter)
            : Bounds(bounds), Filter(std::move(filter)) { };

      // methods
      virtual void AddSample(const Point2f &location, const Sample &sample) = 0;
      virtual void Output() = 0;

      // destructors
      virtual ~AbstractFilm() = default;;

      // data
      const Polytope::Bounds Bounds;
      std::unique_ptr<AbstractFilter> Filter;

   };

}


#endif //POLYTOPE_ABSTRACTFILM_H
