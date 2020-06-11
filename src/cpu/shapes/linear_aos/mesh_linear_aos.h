//
// Created by daniel on 5/3/20.
//

#ifndef POLY_MESH_LINEAR_AOS_H
#define POLY_MESH_LINEAR_AOS_H

#include "../abstract_mesh.h"


namespace poly {
   class MeshLinearAOS : public AbstractMesh {
   public:
      std::vector<Point> Vertices;
      std::vector<Point3ui> Faces;
      std::vector<Normal> Normals;

      MeshLinearAOS(
            const std::shared_ptr<poly::Transform> &objectToWorld,
            const std::shared_ptr<poly::Transform> &worldToObject,
            const std::shared_ptr<poly::Material> &material)
            : AbstractMesh(objectToWorld, worldToObject, material) {}
      ~MeshLinearAOS() override = default;

      void intersect(Ray &ray, Intersection *intersection) override;

      void CalculateVertexNormals() override;

      Point random_surface_point() const override;

      unsigned int CountUniqueVertices();

      void DeduplicateVertices();

      bool Validate();

      /// Removes all faces that have 2 or more identical vertices.
      /// \return The number of faces removed.
      unsigned int RemoveDegenerateFaces();

      /// Removes all vertices that aren't associated with a face.
      /// \return The number of vertices removed.
      unsigned int CountOrphanedVertices();

      void add_vertex(float x, float y, float z) override;

      void add_vertex(Point &v) override;

      void add_packed_face(unsigned int v0, unsigned int v1, unsigned int v2) override;

      void unpack_faces() override;

      Point get_vertex(unsigned int i) const override;

      Point3ui get_vertex_indices_for_face(unsigned int i) const override;
   };
}



#endif //POLY_MESH_LINEAR_AOS_H
