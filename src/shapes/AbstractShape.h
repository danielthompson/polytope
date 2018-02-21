//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLYTOPE_ABSTRACTSHAPE_H
#define POLYTOPE_ABSTRACTSHAPE_H

#include "../structures/Transform.h"
#include "../structures/Ray.h"
#include "../structures/Intersection.h"

namespace Polytope {

   class AbstractShape {
   public:
      // methods
      explicit AbstractShape(const Transform &ObjectToWorld);

      AbstractShape(const Transform &ObjectToWorld, const Transform &WorldToObject);

      virtual bool Hits(Ray &worldSpaceRay) const = 0;
      virtual Intersection Intersect(const Ray &worldSpaceRay) = 0;

      // data
      Transform ObjectToWorld;
      Transform WorldToObject;
   };

}


#endif //POLYTOPE_ABSTRACTSHAPE_H
