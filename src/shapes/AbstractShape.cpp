//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractShape.h"

namespace Polytope {

   using Polytope::Transform;

   AbstractShape::AbstractShape(const Transform &ObjectToWorld, const Material)
      : ObjectToWorld(ObjectToWorld), WorldToObject(ObjectToWorld.Invert()) { }

   AbstractShape::AbstractShape(const Transform &ObjectToWorld, const Transform &WorldToObject)
   : ObjectToWorld(ObjectToWorld), WorldToObject(WorldToObject) { }

}
