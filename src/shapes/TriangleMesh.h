//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLYTOPE_TRIANGLEMESH_H
#define POLYTOPE_TRIANGLEMESH_H

#include "AbstractShape.h"
#include "../shading/Material.h"
#include "../structures/Transform.h"
#include "../structures/Vectors.h"

namespace Polytope {
   class TriangleMesh : public AbstractShape {
   public:
      TriangleMesh(
         const std::shared_ptr<Polytope::Transform> &objectToWorld,
         const std::shared_ptr<Polytope::Transform> &worldToObject,
         const std::shared_ptr<Polytope::Material> &material)
      : AbstractShape(objectToWorld, worldToObject, material) {}

      bool Hits(Ray &worldSpaceRay) const override;
      void CutY(float y);
      void Intersect(Ray &worldSpaceRay, Intersection *intersection) override;

      float SignedDistance(const Point &p, const Point &plane, const Normal &normal);

      Point GetRandomPointOnSurface() override;

      std::vector<Point> Vertices;
      std::vector<Point3ui> Faces;
   };
}

#endif //POLYTOPE_TRIANGLEMESH_H
