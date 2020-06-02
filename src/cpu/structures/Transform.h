//
// Created by Daniel Thompson on 2/19/18.
//

#ifndef POLYTOPE_TRANSFORM_H
#define POLYTOPE_TRANSFORM_H

#include "Vectors.h"
#include "Ray.h"
#include "BoundingBox.h"
#include "Matrix4x4.h"
//#include "../shapes/abstract_mesh.h"

namespace Polytope {

   class Transform {
   public:

      // constructors

      Transform();
      explicit Transform(const float values[4][4]);
      explicit Transform(const Matrix4x4 &matrix);
      Transform(const Matrix4x4 &matrix, const Matrix4x4 &inverse);
      Transform(const Transform &other) = default;

      // operators

      bool operator==(const Transform &rhs) const;
      bool operator!=(const Transform &rhs) const;
      Transform operator*(const Transform &rhs) const;
      Transform &operator*=(const Transform &rhs);

      // methods

      Transform Invert() const;

      void ApplyInPlace(Point &point) const;
      Point Apply(const Point &point) const;
      void ApplyPoint(float &x, float &y, float &z) const;

      void ApplyInPlace(Vector &vector) const;
      Vector Apply(const Vector &vector) const;

      void ApplyInPlace(Normal &normal) const;
      Normal Apply(const Normal &normal) const;

      void ApplyInPlace(Ray &ray) const;
      Ray Apply(const Ray &ray) const;

      void ApplyInPlace(BoundingBox &bb) const;
      BoundingBox Apply(const BoundingBox &bb) const;

      bool HasScale() const;

      static Transform Rotate(float angle, float x, float y, float z);

      static Transform Translate(const Vector &delta);
      static Transform Translate(float x, float y, float z);

      static Transform Scale(const Vector &delta);
      static Transform Scale(float x, float y, float z);
      static Transform Scale(float t);

      static Transform LookAt(const Point &eye, const Point &lookAt, Vector &up, bool right_handed = true);

      static Transform LookAtLeftHanded(const Point &eye, const Point &lookAt, Vector &up);
      
      // data

      Matrix4x4 Matrix;
      Matrix4x4 Inverse;

      static const Vector xDir, yDir, zDir;
   };
}

#endif //POLYTOPE_TRANSFORM_H