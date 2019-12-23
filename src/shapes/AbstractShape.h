//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_ABSTRACTSHAPE_H
#define POLYTOPE_ABSTRACTSHAPE_H

#include "../structures/Transform.h"
#include "../structures/Ray.h"
#include "../structures/Intersection.h"
#include "../shading/Material.h"
#include "../lights/AbstractLight.h"
#include "../lights/ShapeLight.h"
#include "../structures/Vectors.h"

namespace Polytope {

   class BoundingBox {
   public:
      BoundingBox() = default;
      BoundingBox(const Point &min, const Point &max) : p0(min), p1(max) { }
      Point p0, p1;
      bool Hits(const Ray &worldSpaceRay) const;
      BoundingBox Union(const BoundingBox &b) const;
   };

   class AbstractShape {
   protected:

      // constructors
      AbstractShape(
            const Transform &objectToWorld,
            std::shared_ptr<Material> material);
      AbstractShape(
            const Transform &objectToWorld,
            const Transform &worldToObject,
            std::shared_ptr<Material> material);
      AbstractShape(
            const Transform &objectToWorld,
            const Transform &worldToObject,
            std::shared_ptr<Material> material,
            ShapeLight *light);

   public:

      // methods
      virtual bool Hits(Ray &worldSpaceRay) const = 0;
      virtual void Intersect(Ray &worldSpaceRay, Intersection *intersection) = 0;

      virtual Point GetRandomPointOnSurface() = 0;

      bool IsLight() const {
         return (Light);
      }

      // data
      Transform ObjectToWorld;
      Transform WorldToObject;
      std::shared_ptr<Polytope::Material> Material;
      std::unique_ptr<Polytope::BoundingBox> BoundingBox;

      // TODO change to weak to break cycles?
      ShapeLight *Light = nullptr;
   };

}


#endif //POLYTOPE_ABSTRACTSHAPE_H
