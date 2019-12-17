//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLYTOPE_INTERSECTION_H
#define POLYTOPE_INTERSECTION_H

#include <memory>
#include "Vectors.h"

namespace Polytope {

   class AbstractShape; // predeclared to avoid circular header references

   class Intersection {
   public:

      // methods

      Intersection();

      // data

      AbstractShape *Shape = nullptr;
      Point Location;
      Polytope::Normal Normal;
      Vector Tangent1;
      Vector Tangent2;
      bool Hits = false;

      Vector WorldToLocal(const Vector &world) const;
      Vector LocalToWorld(const Vector &local) const;
   };

}


#endif //POLYTOPE_INTERSECTION_H
