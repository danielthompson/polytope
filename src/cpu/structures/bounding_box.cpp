//
// Created by Daniel Thompson on 12/23/19.
//

#include "bounding_box.h"
#include "stats.h"

extern thread_local poly::stats thread_stats;

namespace poly {
   
   bool bounding_box::hits(const poly::ray &world_space_ray, const poly::vector& inverse_direction) const {
      thread_stats.num_bb_intersections++;
      const bool inside = world_space_ray.origin.x > p0.x && world_space_ray.origin.x < p1.x
                          && world_space_ray.origin.y > p0.y && world_space_ray.origin.y < p1.y
                          && world_space_ray.origin.z > p0.z && world_space_ray.origin.z < p1.z;

      if (inside) {
         thread_stats.num_bb_intersections_hit_inside++;
         return true;
      }
      
      float maxBoundFarT = world_space_ray.min_t;
//      float maxBoundFarT = poly::FloatMax;
      float minBoundNearT = 0;

      const float gammaMultiplier = 1 + 2 * poly::Gamma(3);

      // X
      float tNear = (p0.x - world_space_ray.origin.x) * inverse_direction.x;
      float tFar = (p1.x - world_space_ray.origin.x) * inverse_direction.x;

      float swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;
      if (minBoundNearT > maxBoundFarT) {
         thread_stats.num_bb_intersections_miss++;
         return false;
      }

      // Y
      tNear = (p0.y - world_space_ray.origin.y) * inverse_direction.y;
      tFar = (p1.y - world_space_ray.origin.y) * inverse_direction.y;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      if (minBoundNearT > maxBoundFarT) {
         thread_stats.num_bb_intersections_miss++;
         return false;
      }

      // z
      tNear = (p0.z - world_space_ray.origin.z) * inverse_direction.z;
      tFar = (p1.z - world_space_ray.origin.z) * inverse_direction.z;

      swap = tNear;
      tNear = tNear > tFar ? tFar : tNear;
      tFar = swap > tFar ? swap : tFar;

      tFar *= gammaMultiplier;

      minBoundNearT = (tNear > minBoundNearT) ? tNear : minBoundNearT;
      maxBoundFarT = (tFar < maxBoundFarT) ? tFar : maxBoundFarT;

      if (minBoundNearT <= maxBoundFarT) {
         thread_stats.num_bb_intersections_hit_outside++;
         return true;
      }
      else {
         thread_stats.num_bb_intersections_miss++;
         return false;
      }
   }

   bounding_box bounding_box::Union(const bounding_box &b) const {
      poly::point min, max;
      min.x = p0.x < b.p0.x ? p0.x : b.p0.x;
      min.y = p0.y < b.p0.y ? p0.y : b.p0.y;
      min.z = p0.z < b.p0.z ? p0.z : b.p0.z;

      max.x = p1.x > b.p1.x ? p1.x : b.p1.x;
      max.y = p1.y > b.p1.y ? p1.y : b.p1.y;
      max.z = p1.z > b.p1.z ? p1.z : b.p1.z;

      return bounding_box(min, max);
   }

   bounding_box bounding_box::Union(const poly::point &p) const {
      poly::point min = p0, max = p1;

      if (p.x < min.x)
         min.x = p.x;
      if (p.y < min.y)
         min.y = p.y;
      if (p.z < min.z)
         min.z = p.z;

      if (p.x > max.x)
         max.x = p.x;
      if (p.y > max.y)
         max.y = p.y;
      if (p.z > max.z)
         max.z = p.z;

      return bounding_box(min, max);
   }

   void bounding_box::union_in_place(const poly::point &p) {
      if (p.x < p0.x)
         p0.x = p.x;
      if (p.y < p0.y)
         p0.y = p.y;
      if (p.z < p0.z)
         p0.z = p.z;

      if (p.x > p1.x)
         p1.x = p.x;
      if (p.y > p1.y)
         p1.y = p.y;
      if (p.z > p1.z)
         p1.z = p.z;
   }
   
   void bounding_box::union_in_place(const poly::bounding_box &b) {
      p0.x = p0.x < b.p0.x ? p0.x : b.p0.x;
      p0.y = p0.y < b.p0.y ? p0.y : b.p0.y;
      p0.z = p0.z < b.p0.z ? p0.z : b.p0.z;

      p1.x = p1.x > b.p1.x ? p1.x : b.p1.x;
      p1.y = p1.y > b.p1.y ? p1.y : b.p1.y;
      p1.z = p1.z > b.p1.z ? p1.z : b.p1.z;
   }

   float bounding_box::surface_area() const {
      const float x = p1.x - p0.x;
      const float y = p1.y - p0.y;
      const float z = p1.z - p0.z;
      
      return (2 * x * y) + (2 * y * z) + (2 * z * x);  
   }
}
