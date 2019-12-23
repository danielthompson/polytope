//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_ABSTRACTSHAPE_H
#define POLYTOPE_ABSTRACTSHAPE_H

#include "../structures/Vectors.h"
#include "../structures/Ray.h"
#include "../structures/Intersection.h"
#include "../shading/Material.h"
#include "../lights/AbstractLight.h"
#include "../lights/ShapeLight.h"
#include "../structures/BoundingBox.h"

namespace Polytope {

   // forward declaration
   class Transform;

   class AbstractShape {
   protected:

      AbstractShape(
         std::shared_ptr<Polytope::Transform> objectToWorld,
         std::shared_ptr<Polytope::Transform> worldToObject,
            std::shared_ptr<Polytope::Material> material);
      AbstractShape(
            std::shared_ptr<Polytope::Transform> objectToWorld,
            std::shared_ptr<Polytope::Transform> worldToObject,
            std::shared_ptr<Polytope::Material> material,
            ShapeLight *light);

   public:

      // methods
      virtual bool Hits(Polytope::Ray &worldSpaceRay) const = 0;
      virtual void Intersect(Polytope::Ray &worldSpaceRay, Polytope::Intersection *intersection) = 0;

      virtual Point GetRandomPointOnSurface() = 0;

      bool IsLight() const {
         return (Light);
      }

      // data
      std::shared_ptr<Polytope::Transform> ObjectToWorld;
      std::shared_ptr<Polytope::Transform> WorldToObject;
      std::shared_ptr<Polytope::Material> Material;
      std::unique_ptr<Polytope::BoundingBox> BoundingBox;

      // TODO change to weak to break cycles?
      ShapeLight *Light = nullptr;
   };

}

#endif //POLYTOPE_ABSTRACTSHAPE_H
