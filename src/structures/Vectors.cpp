//
// Created by Daniel on 17-Dec-19.
//


#include <cmath>
#include <cassert>
#include "Vectors.h"

namespace Polytope {
   void Normal::Normalize() {
      const float lengthDivisor = 1.0f / std::sqrt(x * x + y * y + z * z);
      x *= lengthDivisor;
      y *= lengthDivisor;
      z *= lengthDivisor;
   }

   Normal Normal::operator*(const float t) const {
      return Normal(x * t, y * t, z * t);
   }

   bool Normal::operator==(const Normal &rhs) const {
      return x == rhs.x && y == rhs.y && z == rhs.z;
   }

   void Normal::Flip() {
      x = -x;
      y = -y;
      z = -z;
   }

   Vector::Vector(const float x, const float y, const float z) : x(x), y(y), z(z) {}

   Vector::Vector(const Normal &n) : x(n.x), y(n.y), z(n.z) { }

   bool Vector::operator==(const Vector &rhs) const {
      return x == rhs.x && y == rhs.y && z == rhs.z;
   }

   bool Vector::operator!=(const Vector &rhs) const {
      return x != rhs.x || y != rhs.y || z != rhs.z;
   }

   float Vector::Dot(const Vector &v) const {
      return v.x * x + v.y * y + v.z * z;
   }

   float Vector::Dot(const Normal &n) const {
      return n.x * x + n.y * y + n.z * z;
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

   float Point::operator[](const Axis axis) const {
      if (axis == Axis::x)
         return x;
      if (axis == Axis::y)
         return y;
      return z;
   }

   float Vector::Length() const {
      return std::sqrt(x * x + y * y + z * z);
   }

   float Vector::LengthSquared() const {
      return x * x + y * y + z * z;
   }

   Vector Vector::operator*(const float t) const {
      return Vector(x * t, y * t, z * t);
   }

   Vector Vector::operator-(const Vector &rhs) const {
      return Vector(x - rhs.x, y - rhs.y, z - rhs.z);
   }

   Vector Vector::operator-() const {
      return Vector(-x, -y, -z);
   }

   Vector Vector::operator+(const Vector &rhs) const {
      return Vector(x + rhs.x, y + rhs.y, z + rhs.z);
   }

   void Vector::Normalize() {
      float lengthDivisor = 1.0f / Length();
      x *= lengthDivisor;
      y *= lengthDivisor;
      z *= lengthDivisor;
   }

   Vector Vector::Cross(const Vector &rhs) const {
      return Vector(y * rhs.z - z * rhs.y,
                    z * rhs.x - x * rhs.z,
                    x * rhs.y - y * rhs.x);
   }

   Vector Vector::Cross(const Normal &rhs) const {
      return Vector(y * rhs.z - z * rhs.y,
                    z * rhs.x - x * rhs.z,
                    x * rhs.y - y * rhs.x);
   }
   
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

   float Point::Dot(const Polytope::Vector &v) const {
      return v.x * x + v.y * y + v.z * z;
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

   Polytope::Vector Point::operator-(const Point &rhs) const {
      return Polytope::Vector(x - rhs.x, y - rhs.y, z - rhs.z);
   }

   Point Point::operator+(const Polytope::Vector &rhs) const {
      return Point(x + rhs.x, y + rhs.y, z + rhs.z);
   }

   Point &Point::operator+=(const Vector &rhs) {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      return *this;
   }

   Point &Point::operator+=(const Normal &rhs) {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      return *this;
   }

   bool Point::operator<(const Point &rhs) const {
      if (x == rhs.x) {
         if (y == rhs.y) {
            return z < rhs.z;
         }
         return y < rhs.y;
      }
      return x < rhs.x;
   }
}

