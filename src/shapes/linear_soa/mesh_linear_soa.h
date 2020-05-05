//
// Created by daniel on 5/2/20.
//

#ifndef POLYTOPE_MESH_LINEAR_SOA_H
#define POLYTOPE_MESH_LINEAR_SOA_H

#include "../abstract_mesh.h"

namespace Polytope {
   class MeshLinearSOA : public AbstractMesh {
   public:
      MeshLinearSOA(
            const std::shared_ptr<Polytope::Transform> &objectToWorld,
            const std::shared_ptr<Polytope::Transform> &worldToObject,
            const std::shared_ptr<Polytope::Material> &material)
            : AbstractMesh(objectToWorld, worldToObject, material) {}
      ~MeshLinearSOA() override = default;


      void intersect(Polytope::Ray &worldSpaceRay, Polytope::Intersection *intersection) override;
      void CalculateVertexNormals() override;

      Point GetRandomPointOnSurface() const override;

      void add_vertex(Point &v) override;
      void add_vertex(float x, float y, float z) override;
      void add_packed_face(unsigned int v0, unsigned int v1, unsigned int v2) override;
      void unpack_faces() override;
      Point get_vertex(unsigned int i) const override;
      Point3ui get_vertex_indices_for_face(unsigned int i) const override;
      
      std::vector<float> x_packed;
      std::vector<float> y_packed;
      std::vector<float> z_packed;

      std::vector<float> nx_packed;
      std::vector<float> ny_packed;
      std::vector<float> nz_packed;

      std::vector<float> x;
      std::vector<float> y;
      std::vector<float> z;

      std::vector<float> nx;
      std::vector<float> ny;
      std::vector<float> nz;

      // given a face index, returns that face's vertex index, etc
      std::vector<unsigned int> fv0;
      std::vector<unsigned int> fv1;
      std::vector<unsigned int> fv2;

   };
}


#endif //POLYTOPE_MESH_LINEAR_SOA_H
