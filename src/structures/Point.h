//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_POINT_H
#define POLYTOPE_POINT_H

#include "Vector.h"

namespace Polytope {

   class Point {
   public:
      float x, y, z;
      // constructors

      Point() : x(0), y(0), z(0) { };
      Point(const float x, const float y, const float z)  : x(x), y(y), z(z) {}
      Point(const Point &p) : x(p.x), y(p.y), z(p.z) { }

      // operators

      bool operator==(const Point &rhs) const;
      bool operator!=(const Point &rhs) const;
      Vector operator-(const Point &rhs) const;
      Point operator+(const Point &rhs) const;
      Point operator+(const Vector &rhs) const;
      float operator[] (int index) const;
      Point &operator+=(const Vector &rhs) {
         x += rhs.x;
         y += rhs.y;
         z += rhs.z;
         return *this;
      }

      Point &operator+=(const Normal &rhs) {
         x += rhs.x;
         y += rhs.y;
         z += rhs.z;
         return *this;
      }

      float Dot(const Point &p) const;
      float Dot(const Vector &v) const;
   };

   class Point3i {
   public:
      int x, y, z;
      Point3i() : x(0), y(0), z(0) { };
      Point3i(const int x, const int y, const int z) : x(x), y(y), z(z) { };
   };

   class Point3ui {
   public:
      unsigned int x, y, z;
      Point3ui() : x(0), y(0), z(0) { };
      Point3ui(const unsigned int x, const unsigned int y, const unsigned int z) : x(x), y(y), z(z) { };
   };
}

#endif //POLYTOPE_POINT_H
