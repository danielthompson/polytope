//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_VECTOR_H
#define POLYTOPE_VECTOR_H

#include "Normal.h"

namespace Polytope {

   class Vector {
   public:

      // constructors

      Vector() : x(0), y(0), z(0) { };
      Vector(float x, float y, float z);
      Vector(const Vector &v);
      Vector(const Normal &n);

      // operators

      bool operator==(const Vector &rhs) const;
      bool operator!=(const Vector &rhs) const;
      float operator[] (int index) const;
      Vector operator*(const float t) const;
      Vector operator-(const Vector &rhs) const;
      Vector operator+(const Vector &rhs) const;
      Vector operator-() const;

      // methods

      float Dot(const Vector &v) const;
      float Dot(const Normal &n) const;
      float Length() const;
      float LengthSquared() const;
      void Normalize();

      // data

      float x, y, z;
   };

}

#endif //POLYTOPE_VECTOR_H
