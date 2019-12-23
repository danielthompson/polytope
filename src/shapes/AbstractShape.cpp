//
// Created by Daniel on 20-Feb-18.
//

#include "AbstractShape.h"
#include <utility>

namespace Polytope {

   //using Polytope::Transform;

   AbstractShape::AbstractShape(
         std::shared_ptr<Polytope::Transform> objectToWorld,
         std::shared_ptr<Polytope::Transform> worldToObject,
         std::shared_ptr<Polytope::Material> material)
      : ObjectToWorld(std::move(objectToWorld)),
        WorldToObject(std::move(worldToObject)),
        Material(std::move(material)),
        BoundingBox(std::make_unique<Polytope::BoundingBox>()) {   }

   AbstractShape::AbstractShape(
      std::shared_ptr<Polytope::Transform> objectToWorld,
      std::shared_ptr<Polytope::Transform> worldToObject,
         std::shared_ptr<Polytope::Material> material,
         ShapeLight *light)
      : ObjectToWorld(std::move(objectToWorld)),
        WorldToObject(std::move(worldToObject)),
        Material(std::move(material)),
        Light(light),
        BoundingBox(std::make_unique<Polytope::BoundingBox>()) { };

}
