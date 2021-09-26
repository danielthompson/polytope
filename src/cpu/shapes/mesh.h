//
// Created by daniel on 5/2/20.
//

#ifndef POLY_MESH_LINEAR_SOA_H
#define POLY_MESH_LINEAR_SOA_H

#include "../structures/transform.h"
#include "../structures/intersection.h"
#include "../shading/Material.h"

namespace poly {

   /**
    * Represents the geometry for a mesh.
    */
   class mesh_geometry {
   public:
      mesh_geometry() = default;

      void add_vertex(point &v);
      void add_vertex(point &v, normal &n);
      void add_vertex(float x, float y, float z);
      void add_vertex(point &p, const float u, const float v);
      void add_vertex(point &p, normal &n, const float u, const float v);
      void add_packed_face(unsigned int v0_index, unsigned int v1_index, unsigned int v2_index);
      void unpack_faces();
      point get_vertex(unsigned int i) const;
      Point3ui get_vertex_indices_for_face(unsigned int i) const;
      void get_vertices_for_face(unsigned int i, poly::point vertices[3]) const;
      
      // vertex locations
      std::vector<float> x;
      std::vector<float> y;
      std::vector<float> z;

      std::vector<float> x_packed;
      std::vector<float> y_packed;
      std::vector<float> z_packed;

      // vertex normals
      bool has_vertex_normals = false;
      std::vector<float> nx;
      std::vector<float> ny;
      std::vector<float> nz;

      std::vector<float> nx_packed;
      std::vector<float> ny_packed;
      std::vector<float> nz_packed;

      // vertex UVs
      bool has_vertex_uvs = false;
      std::vector<float> u;
      std::vector<float> v;
      
      std::vector<float> u_packed;
      std::vector<float> v_packed;
      
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

   /**
    * Represents single instance's data for the given mesh geometry.
    */
   class Mesh {
   public:
      Mesh(const std::shared_ptr<poly::transform> &objectToWorld,
           const std::shared_ptr<poly::transform> &worldToObject,
           const std::shared_ptr<poly::Material> &material,
           const std::shared_ptr<poly::mesh_geometry> &parent) 
           : object_to_world(objectToWorld), world_to_object(worldToObject), material(material), mesh_geometry(parent) { 
         recalculate_bounding_box();
      } 
           
      ~Mesh();
      
      bool hits(const poly::ray& world_ray, const std::vector<unsigned int>& face_indices);
      bool hits(const poly::ray& world_ray, const unsigned int* face_indices, unsigned int num_face_indices) const;
      void intersect(poly::ray& worldSpaceRay, poly::intersection& intersection);
      void intersect(poly::ray& world_ray, poly::intersection& intersection, const std::vector<unsigned int>& face_indices);
      void intersect(poly::ray& world_ray, poly::intersection& intersection, const unsigned int* face_indices, unsigned int num_face_indices) const;

      float surface_area(unsigned int face_index) const;
      
      point random_surface_point() const;

      bool is_light() const {
         return (spd != nullptr);
      }

      void recalculate_bounding_box();
      
      std::shared_ptr<poly::transform> object_to_world;
      std::shared_ptr<poly::transform> world_to_object;
      std::shared_ptr<poly::Material> material;
      std::shared_ptr<poly::bounding_box> bounding_box;
      std::shared_ptr<poly::SpectralPowerDistribution> spd;
      std::shared_ptr<poly::mesh_geometry> mesh_geometry;
      
      poly::bounding_box world_bb;
   };
}

#endif //POLY_MESH_LINEAR_SOA_H
