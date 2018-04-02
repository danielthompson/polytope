//
// Created by dthompson on 20 Feb 18.
//

#include "Intersection.h"

namespace Polytope {

   Intersection::Intersection()
         : Location(Point(0, 0, 0)) { }

   Vector Intersection::WorldToLocal(const Vector &world) const {
      return Vector();
   }

   Vector Intersection::LocalToWorld(const Vector &local) const {
      return Vector();
   }

}