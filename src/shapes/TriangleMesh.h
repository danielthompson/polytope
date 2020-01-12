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
      BVHNode* high = nullptr;
      BVHNode* low = nullptr;
      std::vector<Point3ui> faces;
      BoundingBox bbox;

      void ShrinkBoundingBox(const std::vector<Point> &vertices);
   };

   class TriangleMesh : public AbstractShape {
   public:
      TriangleMesh(
         const std::shared_ptr<Polytope::Transform> &objectToWorld,
         const std::shared_ptr<Polytope::Transform> &worldToObject,
         const std::shared_ptr<Polytope::Material> &material)
      : AbstractShape(objectToWorld, worldToObject, material), root(nullptr) {}

      bool Hits(Ray &worldSpaceRay) const override;
      void SplitX(float x);
      void SplitY(float y);
      void SplitZ(float z);
      void Split(const Point &pointOnPlane, const Normal &normal);
      void Bound();
      void Intersect(Ray &ray, Intersection *intersection) override;
      void IntersectFaces(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces);
      void IntersectNode(Ray &ray, Intersection *intersection, BVHNode* node);

      void CalculateVertexNormals();

      Point GetRandomPointOnSurface() override;

      std::vector<Point> Vertices;
      std::vector<Point3ui> Faces;
      std::vector<Normal> Normals;
      BVHNode* root;

   private:
      void Bound(BVHNode* node, const std::vector<Point3ui> &faces, Axis axis);
      void Split(Axis axis, float split);
   };


}

#endif //POLYTOPE_TRIANGLEMESH_H
