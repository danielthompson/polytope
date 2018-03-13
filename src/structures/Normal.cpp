//
// Created by dthompson on 20 Feb 18.
//

#include <cmath>
#include "Normal.h"

namespace Polytope {

   void Normal::Normalize() {
      float lengthDivisor = 1.0f / Length();
      x *= lengthDivisor;
      y *= lengthDivisor;
      z *= lengthDivisor;
   }

   float Normal::Length() {
      return sqrt(x * x + y * y + z * z);
   }

   float Normal::LengthSquared() {
      return (x * x + y * y + z * z);
   }

   Normal Normal::operator*(const float t) const {
      return Normal(x * t, y * t, z * t);
   }
}