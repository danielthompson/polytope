//
// Created by Daniel on 20-Feb-18.
//

#ifndef POLY_RAY_H
#define POLY_RAY_H

#include "Vectors.h"
#include "../constants.h"

namespace poly {

   class ray {
   public:
      poly::point origin;
      poly::vector direction;
      float min_t = poly::Infinity;
      float max_t = poly::Infinity;

      ray() : min_t(poly::Infinity), max_t(poly::Infinity) { };
      ray(const poly::point &origin, const poly::vector &direction) :
         origin(origin), direction(direction) { }
      ray(float ox, float oy, float oz, float dx, float dy, float dz) :
         origin(ox, oy, oz), direction(dx, dy, dz) { };

      bool operator==(const ray &rhs) const {
         return origin == rhs.origin &&
                direction == rhs.direction &&
                min_t == rhs.min_t &&
                max_t == rhs.max_t;
      }

      bool operator!=(const ray &rhs) const {
         return !(rhs == *this);
      }

      point t(const float t) const {
         return {
               std::fma(direction.x, t, origin.x),
               std::fma(direction.y, t, origin.y),
               std::fma(direction.z, t, origin.z),
         };
      }
   };
}

#endif //POLY_RAY_H
