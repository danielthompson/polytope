//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_ABSTRACTSHAPE_H
#define POLYTOPE_ABSTRACTSHAPE_H

#include "../structures/Transform.h"
#include "../structures/Ray.h"
#include "../structures/Intersection.h"
#include "../shading/Material.h"

namespace Polytope {

   class AbstractShape {
   protected:

      // constructors
      AbstractShape(const Transform &objectToWorld, std::shared_ptr<Material> material);
      AbstractShape(const Transform &objectToWorld, const Transform &worldToObject, std::shared_ptr<Material> material);

   public:

      // methods
      virtual bool Hits(Ray &worldSpaceRay) const = 0;
      virtual void Intersect(const Ray &worldSpaceRay, Intersection *intersection) = 0;

      // data
      Transform ObjectToWorld;
      Transform WorldToObject;
      std::shared_ptr<Material> Material;
   };

}


#endif //POLYTOPE_ABSTRACTSHAPE_H
