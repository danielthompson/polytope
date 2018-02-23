//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_INTERSECTION_H
#define POLYTOPE_INTERSECTION_H

#include <memory>
#include "Point.h"
#include "Normal.h"



namespace Polytope {

   class AbstractShape; // predeclared to avoid circular header references

   class Intersection {
   public:

      // methods

      Intersection();

      // data

      AbstractShape *Shape;
      Point Location;
      Normal Normal;
      bool Hits;
   };

}


#endif //POLYTOPE_INTERSECTION_H
