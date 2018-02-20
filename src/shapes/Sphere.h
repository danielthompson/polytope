//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_SPHERE_H
#define POLYTOPE_SPHERE_H


#include "AbstractShape.h"
#include "../structures/Point.h"
#include "../structures/Transform.h"

namespace Polytope {

   class Sphere : AbstractShape {
   public:
      explicit Sphere(const Transform &transform);

      bool Hits(const Ray &worldSpaceRay) override;
      Intersection Intersect(const Ray &worldSpaceRay) override;

   private:
      const Point Origin = Point(0, 0, 0);
      const float Radius = 1.0f;

   };

}


#endif //POLYTOPE_SPHERE_H
