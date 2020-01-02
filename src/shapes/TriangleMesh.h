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
   class BVHNode {
   public:
      BVHNode* high;
      BVHNode* low;
      std::vector<Point3ui> faces;
      BoundingBox bbox;
   };

   class TriangleMesh : public AbstractShape {
   public:
      TriangleMesh(
         const std::shared_ptr<Polytope::Transform> &objectToWorld,
         const std::shared_ptr<Polytope::Transform> &worldToObject,
         const std::shared_ptr<Polytope::Material> &material)
      : AbstractShape(objectToWorld, worldToObject, material) {}

      bool Hits(Ray &worldSpaceRay) const override;
      void SplitX(float x);
      void SplitY(float y);
      void SplitZ(float z);
      void Split(const Point &pointOnPlane, const Normal &normal);
      void Bound();
      void Intersect(Ray &worldSpaceRay, Intersection *intersection) override;
      void IntersectFaces(Ray &worldSpaceRay, Ray &objectSpaceRay, Intersection *intersection, const std::vector<Point3ui> &faces);
      float GetExtentX();
      float GetExtentY();
      float GetExtentZ();


      Point GetRandomPointOnSurface() override;

      std::vector<Point> Vertices;
      std::vector<Point3ui> Faces;
      BVHNode* root;
   };


}

#endif //POLYTOPE_TRIANGLEMESH_H
