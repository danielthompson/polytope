//
// Created by dthompson on 20 Feb 18.
//

#ifndef POLY_INTERSECTION_H
#define POLY_INTERSECTION_H

#include "Vectors.h"

namespace poly {

   class Mesh;
   
   class intersection {
   public:

      intersection() : location(point(0, 0, 0)), num_bb_hits(0), num_bb_misses(0), num_triangle_isects(0) { }
      
      poly::vector world_to_local(const poly::vector &world) const {
         vector local = vector(world.dot(tangent_1), world.dot(bent_normal), world.dot(tangent_2));
         local.normalize();
         return local;
      }

      poly::vector local_to_world(const poly::vector &local) const {
         poly::vector world = vector(tangent_1.x * local.x + bent_normal.x * local.y + tangent_2.x * local.z,
                                     tangent_1.y * local.x + bent_normal.y * local.y + tangent_2.y * local.z,
                                     tangent_1.z * local.x + bent_normal.z * local.y + tangent_2.z * local.z);

         world.normalize();
         return world;
      }

      const poly::Mesh *shape = nullptr;
      poly::point location;
      poly::vector error;
      poly::normal geo_normal;
      poly::normal bent_normal;
      poly::vector tangent_1;
      poly::vector tangent_2;
      unsigned int face_index;
      unsigned int mesh_index;
      bool Hits = false;
      unsigned int num_bb_hits;
      unsigned int num_bb_misses;
      unsigned int num_triangle_isects;
      float u, v, w;
      float u_tex_lerp, v_tex_lerp;
      int x, y;
   };
}

#endif //POLY_INTERSECTION_H
