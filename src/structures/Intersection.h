//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_INTERSECTION_H
#define POLYTOPE_INTERSECTION_H

#include "Point.h"

namespace Polytope {

   class Intersection {
   public:

      // methods

      Intersection();

      // data

      Point Location;
      bool Hits;
   };

}


#endif //POLYTOPE_INTERSECTION_H
