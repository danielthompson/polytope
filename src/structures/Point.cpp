//
// Created by Daniel Thompson on 2/18/18.
//

#include <assert.h>
#include "Point.h"

namespace Polytope {

   Point::Point(float x, float y, float z) : x(x), y(y), z(z) {}

   Point::Point(const Point &p) = default;

   bool Point::operator==(const Point &rhs) const {
      return x == rhs.x &&
             y == rhs.y &&
             z == rhs.z;
   }

   bool Point::operator!=(const Point &rhs) const {
      return !(rhs == *this);
   }

   float Point::Dot(const Point &p) const {
      return p.x * x + p.y * y + p.z * z;
   }

   float Point::operator[](const int index) const {
      assert(index >= 0);
      assert(index <= 2);
      if (index == 0)
         return x;
      if (index == 1)
         return y;
      return z;
   }

   Point Point::operator+(const Point &rhs) const {
      return Point(x + rhs.x, y + rhs.y, z + rhs.z);
   }

   Vector Point::operator-(const Point &rhs) const {
      return Vector(x - rhs.x, y - rhs.y, z - rhs.z);
   }

}