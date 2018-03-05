//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractShape.h"

#include <utility>

namespace Polytope {

   //using Polytope::Transform;

   AbstractShape::AbstractShape(const Transform &objectToWorld, std::shared_ptr<Polytope::Material> material)
      : ObjectToWorld(objectToWorld), WorldToObject(objectToWorld.Invert()), Material(std::move(material)) { }

   AbstractShape::AbstractShape(const Transform &objectToWorld, const Transform &worldToObject,
                                std::shared_ptr<Polytope::Material> material)
   : ObjectToWorld(objectToWorld), WorldToObject(worldToObject), Material(std::move(material)) {   }
}
