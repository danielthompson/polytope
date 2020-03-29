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
      BVHNode* parent = nullptr;
      std::vector<Point3ui> faces;
      BoundingBox bbox;

      void ShrinkBoundingBox(const std::vector<Point> &vertices, const std::vector<Point3ui> &nodeFaces);
   };

   class TriangleMesh : public AbstractShape {
   public:
      std::vector<Point> Vertices;
      std::vector<Point3ui> Faces;
      std::vector<Normal> Normals;
      BVHNode* root;

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
      void IntersectFacesISPC(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces);
      void IntersectFacesISPC_SOA(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces);
      void IntersectFacesISPC_SOA2(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces);
      void IntersectNode(Ray &ray, Intersection *intersection, BVHNode* node, unsigned int depth);

      void CalculateVertexNormals();

      Point GetRandomPointOnSurface() override;

      unsigned int CountUniqueVertices();

      void DeduplicateVertices();

      bool Validate();

      /// Removes all faces that have 2 or more identical vertices.
      /// \return The number of faces removed.
      unsigned int RemoveDegenerateFaces();

      /// Removes all vertices that aren't associated with a face.
      /// \return The number of vertices removed.
      unsigned int CountOrphanedVertices();

   private:
      void Bound(BVHNode* node, const std::vector<Point3ui> &faces, unsigned int depth);
      void Split(Axis axis, float split);
   };
   
   class TriangleMeshSOA : public AbstractShape {
   public:
      std::vector<float> x;
      std::vector<float> y;
      std::vector<float> z;

      std::vector<float> nx;
      std::vector<float> ny;
      std::vector<float> nz;

      unsigned int num_vertices = 0;
      
      // given a face index, returns that face's vertex index, etc
      std::vector<unsigned int> fv0;
      std::vector<unsigned int> fv1;
      std::vector<unsigned int> fv2;
      
      unsigned int num_faces = 0;
      
      BVHNode* root;
      
      TriangleMeshSOA(
            const std::shared_ptr<Polytope::Transform> &objectToWorld,
            const std::shared_ptr<Polytope::Transform> &worldToObject,
            const std::shared_ptr<Polytope::Material> &material)
            : AbstractShape(objectToWorld, worldToObject, material), root(nullptr) {}

      void CalculateVertexNormals();

      bool Hits(Polytope::Ray &worldSpaceRay) const override;

      void Intersect(Polytope::Ray &worldSpaceRay, Polytope::Intersection *intersection) override;

      Point GetRandomPointOnSurface() override;
   };
}

#endif //POLYTOPE_TRIANGLEMESH_H
