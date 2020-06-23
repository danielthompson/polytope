//
// Created by daniel on 6/20/20.
//

#ifndef POLYTOPE_STATS_H
#define POLYTOPE_STATS_H

namespace poly {
   struct stats {
      unsigned long num_bb_intersections;
      unsigned long num_bb_intersections_hit_inside;
      unsigned long num_bb_intersections_hit_outside;
      unsigned long num_bb_intersections_miss;
      unsigned long num_triangle_intersections;
      unsigned long num_triangle_intersections_hit;
      
      void add(const poly::stats &other) {
         num_bb_intersections += other.num_bb_intersections;
         num_bb_intersections_hit_inside += other.num_bb_intersections_hit_inside;
         num_bb_intersections_hit_outside += other.num_bb_intersections_hit_outside;
         num_bb_intersections_miss += other.num_bb_intersections_miss;
         num_triangle_intersections += other.num_triangle_intersections;
         num_triangle_intersections_hit += other.num_triangle_intersections_hit;
      } 
   };   
}

#endif //POLYTOPE_STATS_H
