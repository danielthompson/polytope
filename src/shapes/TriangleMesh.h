//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLYTOPE_TRIANGLEMESH_H
#define POLYTOPE_TRIANGLEMESH_H

#include "AbstractShape.h"
#include "../shading/Material.h"
#include "../structures/Transform.h"

namespace Polytope {
   class TriangleMesh : public AbstractShape {
   public:
      TriangleMesh(const Transform &objectToWorld, const std::shared_ptr<Polytope::Material> &material)
      : AbstractShape(objectToWorld, material) {}

      bool Hits(Ray &worldSpaceRay) const override {
         return false;
      }

      void Intersect(const Ray &worldSpaceRay, Intersection *intersection) override {

      }

      Point GetRandomPointOnSurface() override {
         return Point();
      }

   public:
      std::vector<Point> Vertices;
      std::vector<Point3ui> Faces;

   };

   class TriangleMeshTemp {
   public:
      std::vector<Point> Vertices;
      std::vector<Point3ui> Faces;

      Transform ObjectToWorld;
      std::shared_ptr<Material> Material;
   };
}


#endif //POLYTOPE_TRIANGLEMESH_H
