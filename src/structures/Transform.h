//
// Created by Daniel Thompson on 2/19/18.
//

#ifndef POLYTOPE_TRANSFORM_H
#define POLYTOPE_TRANSFORM_H

#include "Matrix4x4.h"
#include "Point.h"
#include "Vector.h"
#include "Ray.h"
#include "Normal.h"

namespace Polytope {

   class Transform {
   public:

      // constructors

      Transform();
      explicit Transform(const float values[4][4]);
      explicit Transform(const Matrix4x4 &matrix);
      explicit Transform(const Matrix4x4 &matrix, const Matrix4x4 &inverse);

      // operators

      bool operator==(const Transform &rhs) const;
      bool operator!=(const Transform &rhs) const;
      Transform operator*(const Transform &rhs) const;
      Transform &operator*=(const Transform &rhs);

      // methods

      Transform Invert() const;

      void ApplyInPlace(Point &point) const;
      Point Apply(const Point &point) const;

      void ApplyInPlace(Vector &vector) const;
      Vector Apply(const Vector &vector) const;

      void ApplyInPlace(Normal &normal) const;
      Normal Apply(const Normal &normal) const;

      void ApplyInPlace(Ray &ray) const;
      Ray Apply(const Ray &ray) const;

      bool HasScale() const;

      static Transform Translate(const Vector &delta);
      static Transform Translate(float x, float y, float z);

      static Transform Scale(const Vector &delta);
      static Transform Scale(float x, float y, float z);
      static Transform Scale(float t);

      static Transform LookAt(const Point &eye, const Point &lookAt, Vector &up);

      // data

      Matrix4x4 Matrix;
      Matrix4x4 Inverse;

      static const Vector xDir, yDir, zDir;
   };
}

#endif //POLYTOPE_TRANSFORM_H
