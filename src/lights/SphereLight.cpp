//
// Created by Daniel Thompson on 3/10/18.
//

#include <random>
#include "SphereLight.h"

namespace Polytope {

   Point SphereLight::GetRandomPointOnSurface() const {
      std::random_device rd{};
      std::mt19937 generator {rd()};

      std::normal_distribution<float> distribution{ 0.0f, 1.0f };

      distribution(generator);

      float x, y, z, d2;
      do {
         x = distribution(generator);
         y = distribution(generator);
         z = distribution(generator);
         d2 = x*x + y*y + z*z;
      } while (d2 <= DenormMin);
      float s = sqrt(1.0f / d2);
      return Point(x * s, y * s, z * s);
   }
}