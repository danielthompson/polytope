//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_POINT_H
#define POLYTOPE_POINT_H

#include "Vector.h"

namespace Polytope {

   class Point {
   public:

      // constructors

      Point() : x(0), y(0), z(0) { };
      Point(float x, float y, float z)  : x(x), y(y), z(z) {}
      Point(const Point &p) = default;

      // operators

      bool operator==(const Point &rhs) const;
      bool operator!=(const Point &rhs) const;
      Vector operator-(const Point &rhs) const;
      Point operator+(const Point &rhs) const;
      Point operator+(const Vector &rhs) const;
      float operator[] (int index) const;

      // methods

      float Dot(const Point &p) const;

      // data

      float x, y, z;

   };
}

#endif //POLYTOPE_POINT_H
