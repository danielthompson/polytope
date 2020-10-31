//
// Created by daniel on 5/2/20.
//

#ifndef POLY_MESH_LINEAR_SOA_H
#define POLY_MESH_LINEAR_SOA_H

#include "../structures/Transform.h"
#include "../structures/Intersection.h"
#include "../shading/Material.h"

namespace poly {

   /**
    * Represents the geometry for a mesh.
    */
   class mesh_geometry {
   public:
      mesh_geometry() = default;

      void add_vertex(Point &v);
      void add_vertex(Point &v, Normal &n);
      void add_vertex(float x, float y, float z);
      void add_packed_face(unsigned int v0_index, unsigned int v1_index, unsigned int v2_index);
      void unpack_faces();
      Point get_vertex(unsigned int i) const;
      Point3ui get_vertex_indices_for_face(unsigned int i) const;
      void get_vertices_for_face(unsigned int i, poly::Point vertices[3]) const;
      
      // TODO remove this, each instance needs its own world-space bb
      BoundingBox bb;
      
      // vertexes
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
      Mesh(const std::shared_ptr<poly::Transform> &objectToWorld,
           const std::shared_ptr<poly::Transform> &worldToObject,
           const std::shared_ptr<poly::Material> &material,
           const std::shared_ptr<poly::mesh_geometry> &parent) 
           : object_to_world(objectToWorld), world_to_object(worldToObject), material(material), mesh_geometry(parent) { 
         // TODO calculate world bounding box
      } 
           
      ~Mesh();
      
      bool hits(const poly::Ray& world_ray, const std::vector<unsigned int>& face_indices);
      bool hits(const poly::Ray& world_ray, const unsigned int* face_indices, unsigned int num_face_indices) const;
      void intersect(poly::Ray& worldSpaceRay, poly::Intersection& intersection);
      void intersect(poly::Ray& world_ray, poly::Intersection& intersection, const std::vector<unsigned int>& face_indices);
      void intersect(poly::Ray& world_ray, poly::Intersection& intersection, const unsigned int* face_indices, unsigned int num_face_indices) const;

      float surface_area(unsigned int face_index) const;
      
      Point random_surface_point() const;

      bool is_light() const {
         return (spd != nullptr);
      }

      std::shared_ptr<poly::Transform> object_to_world;
      std::shared_ptr<poly::Transform> world_to_object;
      std::shared_ptr<poly::Material> material;
      std::shared_ptr<poly::BoundingBox> bounding_box;
      std::shared_ptr<poly::SpectralPowerDistribution> spd;
      std::shared_ptr<poly::mesh_geometry> mesh_geometry;
   };
   
   
}


#endif //POLY_MESH_LINEAR_SOA_H
