//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_SPHERE_H
#define POLYTOPE_SPHERE_H


#include "AbstractShape.h"
#include "../structures/Vectors.h"
#include "../structures/Transform.h"

namespace Polytope {

   class Sphere : public AbstractShape {
   public:

      // constructors
      Sphere(const Transform &objectToWorld, std::shared_ptr<Polytope::Material> material);
      Sphere(const Transform &objectToWorld, const Transform &worldToObject, std::shared_ptr<Polytope::Material> material);

      // methods
      bool Hits(Ray &worldSpaceRay) const override;
      void Intersect(Ray &worldSpaceRay, Intersection *intersection) override;
      Point GetRandomPointOnSurface() override;

   private:

      // data
      static const Point Origin;
      static constexpr float Radius = 1.0f;
   };
}

#endif //POLYTOPE_SPHERE_H