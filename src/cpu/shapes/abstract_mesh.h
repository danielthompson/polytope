//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLY_ABSTRACT_MESH_H
#define POLY_ABSTRACT_MESH_H

#include "../structures/Vectors.h"
#include "../structures/Ray.h"
#include "../structures/Intersection.h"
#include "../shading/Material.h"
#include "../structures/BoundingBox.h"
#include "../structures/Transform.h"
#include "../shading/spectrum.h"

namespace poly {

   // forward declaration
   //class Transform;

   class AbstractMesh {
   public:
      AbstractMesh(
            std::shared_ptr<poly::Transform> object_to_world,
            std::shared_ptr<poly::Transform> world_to_object,
            std::shared_ptr<poly::Material> material)
            : ObjectToWorld(std::move(object_to_world)),
              WorldToObject(std::move(world_to_object)),
              Material(std::move(material)),
              BoundingBox(std::make_unique<poly::BoundingBox>()) {   }
      virtual ~AbstractMesh() { }
              
      std::shared_ptr<poly::Transform> ObjectToWorld;
      std::shared_ptr<poly::Transform> WorldToObject;
      std::shared_ptr<poly::Material> Material;
      std::unique_ptr<poly::BoundingBox> BoundingBox;
      std::shared_ptr<poly::SpectralPowerDistribution> spd;

      unsigned int num_vertices_packed = 0;
      unsigned int num_vertices = 0;
      unsigned int num_faces = 0;
      
      virtual void intersect(poly::Ray &world_ray, poly::Intersection *intersection) = 0;
      virtual Point random_surface_point() const = 0;
      bool is_light() const {
         return (spd != nullptr);
      }
      
      virtual void add_vertex(float x, float y, float z) = 0;
      virtual void add_vertex(Point &v) = 0;
      virtual void add_packed_face(unsigned int v0, unsigned int v1, unsigned int v2) = 0;
      virtual void unpack_faces() = 0;

      virtual Point get_vertex(unsigned int i) const = 0;
      virtual Point3ui get_vertex_indices_for_face(unsigned int i) const = 0;
      
      // TODO recalc bounding box
      virtual void Bound() {

      }

      virtual void CalculateVertexNormals() {

      }

      
   };
}

#endif //POLY_ABSTRACT_MESH_H
