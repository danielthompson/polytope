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
   public:
      // methods
      explicit AbstractShape(const Transform &ObjectToWorld, const Material);

      AbstractShape(const Transform &ObjectToWorld, const Transform &WorldToObject);

      virtual bool Hits(Ray &worldSpaceRay) const = 0;
      virtual void Intersect(const Ray &worldSpaceRay, Intersection *intersection) = 0;

      // data
      Transform ObjectToWorld;
      Transform WorldToObject;
      Material Material;
   };

}


#endif //POLYTOPE_ABSTRACTSHAPE_H
