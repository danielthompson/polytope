//
// Created by Daniel Thompson on 3/9/18.
//

#ifndef POLY_GEOMETRYCALCULATIONS_H
#define POLY_GEOMETRYCALCULATIONS_H

#include <random>
#include "../structures/Point.h"
#include "../../cpu/constants.h"
//
//namespace poly {
//
//   class GeometryCalculations {
//   public:
//      static Point GetRandomPointOnSphere() {
//
//         normalDistribution(generator);
//
//         float x, y, z, d2;
//         do {
//            x = normalDistribution(generator);
//            y = normalDistribution(generator);
//            z = normalDistribution(generator);
//            d2 = x*x + y*y + z*z;
//         } while (d2 <= DenormMin);
//         float s = sqrt(1.0f / d2);
//         return Point(x * s, y * s, z * s);
//      }
//
//   private:
//      static std::random_device rd { };
//      static std::mt19937 generator {rd()};
//      static std::normal_distribution<float> normalDistribution{ 0.0f, 1.0f };
//
//   };
//
//}

#endif //POLY_GEOMETRYCALCULATIONS_H
