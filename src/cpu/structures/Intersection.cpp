//
// Created by dthompson on 20 Feb 18.
//

#include "Intersection.h"
#include "Vectors.h"

namespace Polytope {

   Intersection::Intersection()
         : Location(Point(0, 0, 0)) { }

   Vector Intersection::WorldToLocal(const Vector &world) const {
      Vector v = Vector(world.Dot(Tangent1), world.Dot(Normal), world.Dot(Tangent2));
      v.Normalize();
      return v;
   }

   Vector Intersection::LocalToWorld(const Vector &local) const {
      Vector v = Vector(Tangent1.x * local.x + Normal.x * local.y + Tangent2.x * local.z,
                    Tangent1.y * local.x + Normal.y * local.y + Tangent2.y * local.z,
                    Tangent1.z * local.x + Normal.z * local.y + Tangent2.z * local.z);
      
      v.Normalize();
      return v;
   }

}