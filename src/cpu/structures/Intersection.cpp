//
// Created by dthompson on 20 Feb 18.
//

#include "Intersection.h"
#include "Vectors.h"

namespace poly {

   Vector Intersection::WorldToLocal(const Vector &world) const {
      Vector v = Vector(world.Dot(Tangent1), world.Dot(bent_normal), world.Dot(Tangent2));
      v.Normalize();
      return v;
   }

   Vector Intersection::LocalToWorld(const Vector &local) const {
      Vector v = Vector(Tangent1.x * local.x + bent_normal.x * local.y + Tangent2.x * local.z,
                    Tangent1.y * local.x + bent_normal.y * local.y + Tangent2.y * local.z,
                    Tangent1.z * local.x + bent_normal.z * local.y + Tangent2.z * local.z);
      
      v.Normalize();
      return v;
   }
}
