//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "transform.h"
#include "Vectors.h"
#include "bounding_box.h"

namespace poly {

   const poly::vector transform::x_dir = vector(1, 0, 0);
   const poly::vector transform::y_dir = vector(0, 1, 0);
   const poly::vector transform::z_dir = vector(0, 0, 1);

   transform::transform() : matrix(poly::matrix()), inverse(poly::matrix()) { }

   transform::transform(const float values[4][4]) : matrix(values) {
      inverse = matrix.inverse();
   }

   transform::transform(float m00, float m01, float m02, float m03,
                        float m10, float m11, float m12, float m13,
                        float m20, float m21, float m22, float m23,
                        float m30, float m31, float m32, float m33)
                      : matrix(m00, m01, m02, m03,
                               m10, m11, m12, m13,
                               m20, m21, m22, m23,
                               m30, m31, m32, m33) {
      inverse = matrix.inverse();
   }

   poly::transform::transform(const poly::matrix &matrix) : matrix(matrix) {
      inverse = this->matrix.inverse();
   }

   poly::transform::transform(const poly::matrix &matrix, const poly::matrix &inverse) : matrix(matrix), inverse(inverse) { }

   poly::transform transform::invert() const {
      return { inverse, matrix };
   }

   bool transform::operator==(const poly::transform &rhs) const {
      return matrix == rhs.matrix &&
             inverse == rhs.inverse;
   }

   bool transform::operator!=(const poly::transform &rhs) const {
      return !(rhs == *this);
   }

   poly::transform transform::operator*(const poly::transform &rhs) const {
      return { this->matrix * rhs.matrix, rhs.inverse * this->inverse };
   }

   // TODO
   poly::transform &transform::operator*=(const poly::transform &rhs) {

      this->matrix *= rhs.matrix;
      this->inverse *= rhs.inverse;

      return *this;
   }

   void transform::apply_in_place(poly::point &point) const {
      float x = point.x;
      float y = point.y;
      float z = point.z;

      float newX = sum_of_products(x, matrix.mat[0][0], y, matrix.mat[0][1]) + std::fmaf(z, matrix.mat[0][2], matrix.mat[0][3]);
      float newY = sum_of_products(x, matrix.mat[1][0], y, matrix.mat[1][1]) + std::fmaf(z, matrix.mat[1][2], matrix.mat[1][3]);
      float newZ = sum_of_products(x, matrix.mat[2][0], y, matrix.mat[2][1]) + std::fmaf(z, matrix.mat[2][2], matrix.mat[2][3]);

      float w = sum_of_products(x, matrix.mat[3][0], y, matrix.mat[3][1]) + std::fmaf(z, matrix.mat[3][2], matrix.mat[3][3]);

      if (w == 1) {
         point.x = newX;
         point.y = newY;
         point.z = newZ;
      }
      else {
         float divisor = 1.f / w;
         point.x = newX * divisor;
         point.y = newY * divisor;
         point.z = newZ * divisor;
      }
   }

   poly::point transform::apply(const poly::point &point) const {
      float x = point.x;
      float y = point.y;
      float z = point.z;

      float newX = x * matrix.mat[0][0] + y * matrix.mat[0][1] + z * matrix.mat[0][2] + matrix.mat[0][3];
      float newY = x * matrix.mat[1][0] + y * matrix.mat[1][1] + z * matrix.mat[1][2] + matrix.mat[1][3];
      float newZ = x * matrix.mat[2][0] + y * matrix.mat[2][1] + z * matrix.mat[2][2] + matrix.mat[2][3];

      float w = x * matrix.mat[3][0] + y * matrix.mat[3][1] + z * matrix.mat[3][2] + matrix.mat[3][3];

      if (w == 1) {
         return { newX, newY, newZ };
      }
      else {
         float divisor = 1.f / w;
         return { newX * divisor, newY * divisor, newZ * divisor };
      }
   }

   void transform::apply_in_place(poly::vector &vector) const {
      vector.x = vector.x * matrix.mat[0][0] + vector.y * matrix.mat[0][1] + vector.z * matrix.mat[0][2];
      vector.y = vector.x * matrix.mat[1][0] + vector.y * matrix.mat[1][1] + vector.z * matrix.mat[1][2];
      vector.z = vector.x * matrix.mat[2][0] + vector.y * matrix.mat[2][1] + vector.z * matrix.mat[2][2];
   }

   poly::vector transform::apply(const poly::vector &vector) const {
      return {
            vector.x * matrix.mat[0][0] + vector.y * matrix.mat[0][1] + vector.z * matrix.mat[0][2],
            vector.x * matrix.mat[1][0] + vector.y * matrix.mat[1][1] + vector.z * matrix.mat[1][2],
            vector.x * matrix.mat[2][0] + vector.y * matrix.mat[2][1] + vector.z * matrix.mat[2][2]
      };
   }

   void transform::apply_in_place(normal &normal) const {
      normal.x = normal.x * inverse.mat[0][0] + normal.y * inverse.mat[1][0] + normal.z * inverse.mat[2][0];
      normal.y = normal.x * inverse.mat[0][1] + normal.y * inverse.mat[1][1] + normal.z * inverse.mat[2][1];
      normal.z = normal.x * inverse.mat[0][2] + normal.y * inverse.mat[1][2] + normal.z * inverse.mat[2][2];
   }

   poly::normal transform::Apply(const poly::normal &n) const {
      poly::normal normal(
         n.x * inverse.mat[0][0] + n.y * inverse.mat[1][0] + n.z * inverse.mat[2][0],
         n.x * inverse.mat[0][1] + n.y * inverse.mat[1][1] + n.z * inverse.mat[2][1],
         n.x * inverse.mat[0][2] + n.y * inverse.mat[1][2] + n.z * inverse.mat[2][2]);

      normal.normalize();

      return normal;
   }

   void transform::apply_in_place(poly::ray &ray) const {
      apply_in_place(ray.origin);
      apply_in_place(ray.direction);
   }

   poly::ray transform::apply(const poly::ray &ray) const {
      return {(*this).apply(ray.origin), (*this).apply(ray.direction) };
   }

   void transform::ApplyInPlace(poly::bounding_box &bb) const {
      poly::point p0 = apply(poly::point(bb.p0.x, bb.p0.y, bb.p0.z));
      poly::point p1 = apply(poly::point(bb.p0.x, bb.p0.y, bb.p1.z));
      poly::point p2 = apply(poly::point(bb.p0.x, bb.p1.y, bb.p0.z));
      poly::point p3 = apply(poly::point(bb.p0.x, bb.p1.y, bb.p1.z));
      poly::point p4 = apply(poly::point(bb.p1.x, bb.p0.y, bb.p0.z));
      poly::point p5 = apply(poly::point(bb.p1.x, bb.p0.y, bb.p1.z));
      poly::point p6 = apply(poly::point(bb.p1.x, bb.p1.y, bb.p0.z));
      poly::point p7 = apply(poly::point(bb.p1.x, bb.p1.y, bb.p1.z));

      bb.p0 = p0;
      bb.p1 = p1;

//      bb.UnionInPlace(p0);
//      bb.UnionInPlace(p1);
      bb.union_in_place(p2);
      bb.union_in_place(p3);
      bb.union_in_place(p4);
      bb.union_in_place(p5);
      bb.union_in_place(p6);
      bb.union_in_place(p7);
   }

   poly::bounding_box transform::Apply(const poly::bounding_box &bb) const {
      return bounding_box();
   }

   bool transform::has_scale() const {
      float lengthX = apply(x_dir).length_squared();
      float lengthY = apply(y_dir).length_squared();
      float lengthZ = apply(z_dir).length_squared();

      return (lengthX < .999 || lengthX > 1.001
              || lengthY < .999 || lengthY > 1.001
              || lengthZ < .999 || lengthZ > 1.001);
   }

   poly::transform transform::rotate(const float angle, const float x, const float y, const float z) {
      const float sin = std::sin(angle);
      const float cos = std::cos(angle);
      const float oneMinusCos = (1 - cos);
      poly::matrix matrix(
            cos + x * x * oneMinusCos,     x * y * oneMinusCos - z * sin, x * z * oneMinusCos + y * sin, 0,
            y * x * oneMinusCos + z * sin, cos + y * y * oneMinusCos,     y * z * oneMinusCos - x * sin, 0,
            z * x * oneMinusCos - y * sin, z * y * oneMinusCos + x * sin, cos + z * z * oneMinusCos,     0,
            0,                             0,                             0,                             1
            );

      return poly::transform(matrix);
   }

   poly::transform transform::translate(const poly::vector &delta) {
      poly::matrix matrix(
            1, 0, 0, delta.x,
            0, 1, 0, delta.y,
            0, 0, 1, delta.z,
            0, 0, 0, 1);

      poly::matrix inverse(
            1, 0, 0, -delta.x,
            0, 1, 0, -delta.y,
            0, 0, 1, -delta.z,
            0, 0, 0, 1);

      return { matrix, inverse };
   }

   poly::transform transform::translate(const float x, const float y, const float z)  {
      poly::matrix matrix(
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1);

      poly::matrix inverse(
            1, 0, 0, -x,
            0, 1, 0, -y,
            0, 0, 1, -z,
            0, 0, 0, 1);

      return { matrix, inverse };
   }

   poly::transform transform::scale(const poly::vector &delta) {
      poly::matrix matrix(
            delta.x, 0, 0, 0,
            0, delta.y, 0, 0,
            0, 0, delta.z, 0,
            0, 0, 0, 1);

      poly::matrix inverse(
            1.f/delta.x, 0,    0,    0,
            0,    1.f/delta.y, 0,    0,
            0,    0,    1.f/delta.z, 0,
            0,    0,    0,    1);

      return { matrix, inverse };
   }

   poly::transform transform::scale(const float x, const float y, const float z) {
      poly::matrix matrix(
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1);

      poly::matrix inverse(
            1.f/x, 0,    0,    0,
            0,    1.f/y, 0,    0,
            0,    0,    1.f/z, 0,
            0,    0,    0,    1);

      return { matrix, inverse };
   }

   poly::transform transform::scale(const float t) {
      poly::matrix matrix(
            t, 0, 0, 0,
            0, t, 0, 0,
            0, 0, t, 0,
            0, 0, 0, 1);

      const float oneOverT = 1.0f / t;

      poly::matrix inverse(
            oneOverT, 0,    0,    0,
            0,    oneOverT, 0,    0,
            0,    0,    oneOverT, 0,
            0,    0,    0,    1);

      return { matrix, inverse };
   }

   poly::transform transform::look_at(const poly::point &eye, const poly::point &look_at, poly::vector &up, const bool right_handed) {
      float m[4][4];

      m[0][3] = eye.x;
      m[1][3] = eye.y;
      m[2][3] = eye.z;
      m[3][3] = 1;

      poly::vector dir = look_at - eye;
      dir.normalize();

      up.normalize();
      poly::vector left = up.cross(dir);

      if (left.length() == 0) {
         std::cout << "Bad Transform::LookAt() call - left poly::vector is 0. Up and viewing direction are poly::pointing in the same direction. Using identity.";
         return transform();
      }

      left.normalize();

      const poly::vector newUp = dir.cross(left);

      m[0][0] = left.x;
      m[1][0] = left.y;
      m[2][0] = left.z;
      m[3][0] = 0;

      m[0][1] = newUp.x;
      m[1][1] = newUp.y;
      m[2][1] = newUp.z;
      m[3][1] = 0;

      m[0][2] = dir.x;
      m[1][2] = dir.y;
      m[2][2] = right_handed ? -dir.z : dir.z;
      m[3][2] = 0;

      poly::matrix matrix(m);
      return { matrix, matrix.inverse() };
   }

   poly::transform transform::look_at_left_handed(const poly::point &eye, const poly::point &look_at, poly::vector &up) {

      // need to flip the z-coordinate, since polytope is right-handed internally
      const poly::point fixedEye(eye.x, eye.y, -eye.z);
      const poly::point fixedLookAt(look_at.x, look_at.y, look_at.z);
      poly::vector fixedUp(up.x, up.y, -up.z);

      float m[4][4];

      m[0][3] = fixedEye.x;
      m[1][3] = fixedEye.y;
      m[2][3] = fixedEye.z;
      m[3][3] = 1;

      poly::vector dir = fixedLookAt - fixedEye;
      dir.normalize();

      fixedUp.normalize();
      poly::vector left = fixedUp.cross(dir);

      if (left.length() == 0) {

         std::cout << "Bad Transform::LookAt() call - left poly::vector is 0. Up and viewing direction are poly::pointing in the same direction. Using identity.";
         return transform();

      }

      left.normalize();

      const poly::vector newUp = dir.cross(left);

      m[0][0] = left.x;
      m[1][0] = left.y;
      m[2][0] = left.z;
      m[3][0] = 0;

      m[0][1] = newUp.x;
      m[1][1] = newUp.y;
      m[2][1] = newUp.z;
      m[3][1] = 0;

      m[0][2] = dir.x;
      m[1][2] = dir.y;
      m[2][2] = dir.z;
      m[3][2] = 0;

      poly::matrix matrix(m);
      return { matrix, matrix.inverse() };
   }
}
