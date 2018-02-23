//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_INTERSECTION_H
#define POLYTOPE_INTERSECTION_H

#include <memory>
#include "Point.h"
#include "../shapes/AbstractShape.h"

namespace Polytope {

   class Intersection {
   public:

      // methods

      Intersection();

      // data

      std::shared_ptr<AbstractShape> Shape;
      Point Location;
      Normal Normal;
      bool Hits;
   };

}


#endif //POLYTOPE_INTERSECTION_H
