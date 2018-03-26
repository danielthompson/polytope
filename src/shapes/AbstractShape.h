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

namespace Polytope {

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
            AbstractLight *light);

   public:

      // methods
      virtual bool Hits(Ray &worldSpaceRay) const = 0;
      virtual void Intersect(const Ray &worldSpaceRay, Intersection *intersection) = 0;

      bool IsLight() const {
         return (Light != nullptr);
      }

      // data
      Transform ObjectToWorld;
      Transform WorldToObject;
      std::shared_ptr<Polytope::Material> Material;

      // TODO change to weak to break cycles?
      AbstractLight *Light;
   };

}


#endif //POLYTOPE_ABSTRACTSHAPE_H
