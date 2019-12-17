//
// Created by Daniel on 17-Dec-19.
//

#include <cmath>
#include <cassert>
#include "Vectors.h"

namespace Polytope {
   void Normal::Normalize() {
      float lengthDivisor = 1.0f / Length();
      x *= lengthDivisor;
      y *= lengthDivisor;
      z *= lengthDivisor;
   }

   float Normal::Length() {
      return std::sqrt(x * x + y * y + z * z);
   }

   float Normal::LengthSquared() {
      return (x * x + y * y + z * z);
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

   Vector::Vector(float x, float y, float z) : x(x), y(y), z(z) {}

   Vector::Vector(const Vector &v) : x(v.x), y(v.y), z(v.z) { }

   Vector::Vector(const Normal &n) : x(n.x), y(n.y), z(n.z) { }

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
}