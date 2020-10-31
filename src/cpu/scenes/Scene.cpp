//
// Created by Daniel on 20-Feb-18.
//

#include <cassert>
#include <cmath>
#include "Scene.h"
#include "../constants.h"

namespace poly {

   Intersection Scene::intersect(Ray &ray, int x, int y) {
//      Ray ray2 = ray;
//      Intersection linear_intersection = GetNearestShapeIteratively(this->Shapes, ray);
//      return linear_intersection;

      Intersection bvh_intersection;
      bvh_root.intersect_compact(ray, bvh_intersection);
//      bvh_root.intersect(ray, bvh_intersection);
      return bvh_intersection;

      //      if (linear_intersection.Hits != bvh_intersection.Hits)
//         assert(linear_intersection.Hits == bvh_intersection.Hits);
//      if (linear_intersection.Hits) {
//         assert(linear_intersection.faceIndex == bvh_intersection.faceIndex);
//         assert(linear_intersection.Location == bvh_intersection.Location);
//      }

   }
}
