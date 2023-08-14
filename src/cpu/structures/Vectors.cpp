//
// Created by Daniel on 17-Dec-19.
//


#include <cmath>
#include <cassert>
#include "Vectors.h"

namespace poly {
   void normal::normalize() {
      const float lengthDivisor = 1.0f / std::sqrt(x * x + y * y + z * z);
      x *= lengthDivisor;
      y *= lengthDivisor;
      z *= lengthDivisor;
   }

   normal normal::operator*(const float t) const {
      return normal(x * t, y * t, z * t);
   }

   bool normal::operator==(const normal &rhs) const {
      return x == rhs.x && y == rhs.y && z == rhs.z;
   }

   void normal::flip() {
      x = -x;
      y = -y;
      z = -z;
   }



   bool vector::operator==(const vector &rhs) const {
      return x == rhs.x && y == rhs.y && z == rhs.z;
   }

   bool vector::operator!=(const vector &rhs) const {
      return x != rhs.x || y != rhs.y || z != rhs.z;
   }

   float vector::dot(const vector &v) const {
      return v.x * x + v.y * y + v.z * z;
   }

   float vector::dot(const normal &n) const {
      return n.x * x + n.y * y + n.z * z;
   }

   float vector::operator[](const int index) const {
      assert(index >= 0);
      assert(index <= 2);
      if (index == 0)
         return x;
      if (index == 1)
         return y;
      return z;
   }

   float &point::operator[](const axis axis) {
      if (axis == axis::x)
         return x;
      if (axis == axis::y)
         return y;
      return z;
   }

   float vector::length() const {
      return std::sqrt(x * x + y * y + z * z);
   }

   float vector::length_squared() const {
      return x * x + y * y + z * z;
   }

   vector vector::operator*(const float t) const {
      return vector(x * t, y * t, z * t);
   }

   vector vector::operator-(const vector &rhs) const {
      return vector(x - rhs.x, y - rhs.y, z - rhs.z);
   }

   vector vector::operator-() const {
      return vector(-x, -y, -z);
   }

   vector vector::operator+(const vector &rhs) const {
      return vector(x + rhs.x, y + rhs.y, z + rhs.z);
   }

   void vector::normalize() {
      float lengthDivisor = 1.0f / length();
      x *= lengthDivisor;
      y *= lengthDivisor;
      z *= lengthDivisor;
   }

   vector vector::cross(const vector &rhs) const {
      return vector(y * rhs.z - z * rhs.y,
                    z * rhs.x - x * rhs.z,
                    x * rhs.y - y * rhs.x);
   }

   vector vector::cross(const normal &rhs) const {
      return vector(y * rhs.z - z * rhs.y,
                    z * rhs.x - x * rhs.z,
                    x * rhs.y - y * rhs.x);
   }
   
   bool point::operator==(const point &rhs) const {
      return x == rhs.x &&
             y == rhs.y &&
             z == rhs.z;
   }

   bool point::operator!=(const point &rhs) const {
      return !(rhs == *this);
   }

   float point::dot(const point &p) const {
      return p.x * x + p.y * y + p.z * z;
   }

   float point::dot(const poly::vector &v) const {
      return v.x * x + v.y * y + v.z * z;
   }

   float& point::operator[](const int index) {
      assert(index >= 0);
      assert(index <= 2);
      if (index == 0)
         return x;
      if (index == 1)
         return y;
      return z;
   }

   point point::operator+(const point &rhs) const {
      return point(x + rhs.x, y + rhs.y, z + rhs.z);
   }

   poly::vector point::operator-(const point &rhs) const {
      return poly::vector(x - rhs.x, y - rhs.y, z - rhs.z);
   }

   point point::operator+(const poly::vector &rhs) const {
      return point(x + rhs.x, y + rhs.y, z + rhs.z);
   }

   point &point::operator+=(const vector &rhs) {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      return *this;
   }

   point &point::operator+=(const normal &rhs) {
      x += rhs.x;
      y += rhs.y;
      z += rhs.z;
      return *this;
   }

   bool point::operator<(const point &rhs) const {
      if (x == rhs.x) {
         if (y == rhs.y) {
            return z < rhs.z;
         }
         return y < rhs.y;
      }
      return x < rhs.x;
   }
}

