//
// Created by daniel on 5/3/20.
//

#ifndef POLYTOPE_MESH_BVH_SPLIT_H
#define POLYTOPE_MESH_BVH_SPLIT_H

#include "../abstract_mesh.h"
#include "../../scenes/BVHNode.h"

namespace Polytope {
   class MeshBVHSplit : public AbstractMesh {
   public:
      MeshBVHSplit(
            const std::shared_ptr<Polytope::Transform> &objectToWorld,
            const std::shared_ptr<Polytope::Transform> &worldToObject,
            const std::shared_ptr<Polytope::Material> &material)
            : AbstractMesh(objectToWorld, worldToObject, material), root(nullptr) {}
      ~MeshBVHSplit() override = default;

      void SplitX(float x);
      void SplitY(float y);
      void SplitZ(float z);
      void Split(const Point &pointOnPlane, const Normal &normal);
      void Bound();
      void intersect(Ray &ray, Intersection *intersection) override;
      void IntersectFaces(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces);
      void IntersectFacesISPC(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces);
      void IntersectFacesISPC_SOA(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces);
      void IntersectFacesISPC_SOA2(Ray &ray, Intersection *intersection, const std::vector<Point3ui> &faces);
      void IntersectNode(Ray &ray, Intersection *intersection, BVHNode* node, unsigned int depth);

      void CalculateVertexNormals();

      Point GetRandomPointOnSurface() const override;

      unsigned int CountUniqueVertices() const ;

      void DeduplicateVertices();

      bool Validate() const ;

      /// Removes all faces that have 2 or more identical vertices.
      /// \return The number of faces removed.
      unsigned int RemoveDegenerateFaces();

      /// Removes all vertices that aren't associated with a face.
      /// \return The number of vertices removed.
      unsigned int CountOrphanedVertices() const ;

      void add_vertex(float x, float y, float z) override;
      void add_vertex(Point &v) override;
      void add_packed_face(unsigned int v0, unsigned int v1, unsigned int v2) override;
      void unpack_faces() override;
      Point get_vertex(unsigned int i) const override;
      Point3ui get_vertex_indices_for_face(unsigned int i) const override;

      std::vector<Point> Vertices;
      std::vector<Point3ui> Faces;
      std::vector<Normal> Normals;
      BVHNode* root;

   private:
      void Bound(BVHNode* node, const std::vector<Point3ui> &faces, unsigned int depth);
      void Split(Axis axis, float split);
   };
}


#endif //POLYTOPE_MESH_BVH_SPLIT_H
