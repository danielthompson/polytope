//
// Created by Daniel Thompson on 2/18/18.
//

#include "point.h"

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



