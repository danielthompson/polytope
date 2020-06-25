//
// Created by daniel on 5/2/20.
//

#ifndef POLY_MESH_LINEAR_SOA_H
#define POLY_MESH_LINEAR_SOA_H

#include "../structures/Transform.h"
#include "../structures/Intersection.h"
#include "../shading/Material.h"

namespace poly {
   class Mesh {
   public:
      Mesh(const std::shared_ptr<poly::Transform> &objectToWorld,
           const std::shared_ptr<poly::Transform> &worldToObject,
           const std::shared_ptr<poly::Material> &material) 
           : ObjectToWorld(objectToWorld),
             WorldToObject(worldToObject),
             Material(material),
             BoundingBox(std::make_unique<poly::BoundingBox>()) { }
      ~Mesh();
      
      bool hits(const poly::Ray& world_ray, const std::vector<unsigned int>& face_indices);
      bool hits(const poly::Ray& world_ray, const unsigned int* face_indices, unsigned int num_face_indices) const;
      void intersect(poly::Ray& worldSpaceRay, poly::Intersection& intersection);
      void intersect(poly::Ray& world_ray, poly::Intersection& intersection, const std::vector<unsigned int>& face_indices);
      void intersect(poly::Ray& world_ray, poly::Intersection& intersection, const unsigned int* face_indices, unsigned int num_face_indices) const;
      void CalculateVertexNormals();

      float surface_area(unsigned int face_index) const;
      
      Point random_surface_point() const;

      void add_vertex(Point &v);
      void add_vertex(Point &v, Normal &n);
      void add_vertex(float x, float y, float z);
      void add_packed_face(unsigned int v0_index, unsigned int v1_index, unsigned int v2_index);
      void unpack_faces();
      Point get_vertex(unsigned int i) const;
      Point3ui get_vertex_indices_for_face(unsigned int i) const;

      bool is_light() const {
         return (spd != nullptr);
      }

      std::shared_ptr<poly::Transform> ObjectToWorld;
      std::shared_ptr<poly::Transform> WorldToObject;
      std::shared_ptr<poly::Material> Material;
      std::unique_ptr<poly::BoundingBox> BoundingBox;
      std::shared_ptr<poly::SpectralPowerDistribution> spd;
      
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
      
      // given a face index, returns that face's normal
      std::vector<float> fnx;
      std::vector<float> fny;
      std::vector<float> fnz;

      unsigned int num_vertices_packed = 0;
      unsigned int num_vertices = 0;
      unsigned int num_faces = 0;
      
   };
}


#endif //POLY_MESH_LINEAR_SOA_H
