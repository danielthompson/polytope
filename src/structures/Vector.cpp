//
// Created by Daniel Thompson on 2/18/18.
//

#include <cassert>
#include <cmath>
#include "Vector.h"

namespace Polytope {

   Vector::Vector(float x, float y, float z) : x(x), y(y), z(z) {}

   Vector::Vector(const Vector &v) = default;

   bool Vector::operator==(const Vector &rhs) const {
      return x == rhs.x &&
             y == rhs.y &&
             z == rhs.z;
   }

   bool Vector::operator!=(const Vector &rhs) const {
      return !(rhs == *this);
   }

   float Vector::Dot(const Vector &v) const {
      return v.x * x + v.y * y + v.z * z;
   }

   float Vector::operator[](const int index) const {
      assert(index >= 0);
      assert(index <= 2);
      if (index == 0)
         return x;
      if (index == 1)
         return y;
      return z;
   }

   float Vector::Length() const {
      return std::sqrt(x * x + y * y + z + z);
   }

   float Vector::LengthSquared() const {
      return x * x + y * y + z * z;
   }

   Vector Vector::operator*(const float t) const {
      return Vector(x * t, y * t, z * t);
   }

}