//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "Transform.h"
#include "Vectors.h"
#include "BoundingBox.h"

namespace poly {

   const Vector Transform::xDir = Vector(1, 0, 0);
   const Vector Transform::yDir = Vector(0, 1, 0);
   const Vector Transform::zDir = Vector(0, 0, 1);

   Transform::Transform() : Matrix(Matrix4x4()), Inverse(Matrix4x4()) { }

   Transform::Transform(const float values[4][4]) : Matrix(Matrix4x4(values)) {
      Inverse = Matrix.Inverse();
   }

   Transform::Transform(float m00, float m01, float m02, float m03,
                      float m10, float m11, float m12, float m13,
                      float m20, float m21, float m22, float m23,
                      float m30, float m31, float m32, float m33)
                      : Matrix(m00, m01, m02, m03,
                                  m10, m11, m12, m13,
                                  m20, m21, m22, m23,
                                  m30, m31, m32, m33) {
      Inverse = Matrix.Inverse();
   }

   Transform::Transform(const Matrix4x4 &matrix) : Matrix(matrix) {
      Inverse = Matrix.Inverse();
   }

   Transform::Transform(const Matrix4x4 &matrix, const Matrix4x4 &inverse) : Matrix(matrix), Inverse(inverse) { }

   Transform Transform::Invert() const {
      Transform transform = Transform(Inverse, Matrix);
      return transform;
   }

   bool Transform::operator==(const Transform &rhs) const {
      return Matrix == rhs.Matrix &&
             Inverse == rhs.Inverse;
   }

   bool Transform::operator!=(const Transform &rhs) const {
      return !(rhs == *this);
   }

   Transform Transform::operator*(const Transform &rhs) const {
      Matrix4x4 matrix = this->Matrix * rhs.Matrix;
      Matrix4x4 inverse = rhs.Inverse * this->Inverse;

      return Transform(matrix, inverse);
   }

   // TODO
   Transform &Transform::operator*=(const Transform &rhs) {

      this->Matrix *= rhs.Matrix;
      this->Inverse *= rhs.Inverse;

      return *this;
   }

   void Transform::ApplyInPlace(Point &p) const {
      float x = p.x;
      float y = p.y;
      float z = p.z;

      float newX = sum_of_products(x, Matrix.Matrix[0][0], y, Matrix.Matrix[0][1]) + std::fmaf(z, Matrix.Matrix[0][2], Matrix.Matrix[0][3]);
      float newY = sum_of_products(x, Matrix.Matrix[1][0], y, Matrix.Matrix[1][1]) + std::fmaf(z, Matrix.Matrix[1][2], Matrix.Matrix[1][3]);
      float newZ = sum_of_products(x, Matrix.Matrix[2][0], y, Matrix.Matrix[2][1]) + std::fmaf(z, Matrix.Matrix[2][2], Matrix.Matrix[2][3]);

      float w = sum_of_products(x, Matrix.Matrix[3][0], y, Matrix.Matrix[3][1]) + std::fmaf(z, Matrix.Matrix[3][2], Matrix.Matrix[3][3]);

      if (w == 1) {
         p.x = newX;
         p.y = newY;
         p.z = newZ;
      }
      else {
         float divisor = 1.f / w;
         p.x = newX * divisor;
         p.y = newY * divisor;
         p.z = newZ * divisor;
      }
   }

   Point Transform::Apply(const Point &p) const {
      float x = p.x;
      float y = p.y;
      float z = p.z;

      float newX = x * Matrix.Matrix[0][0] + y * Matrix.Matrix[0][1] + z * Matrix.Matrix[0][2] + Matrix.Matrix[0][3];
      float newY = x * Matrix.Matrix[1][0] + y * Matrix.Matrix[1][1] + z * Matrix.Matrix[1][2] + Matrix.Matrix[1][3];
      float newZ = x * Matrix.Matrix[2][0] + y * Matrix.Matrix[2][1] + z * Matrix.Matrix[2][2] + Matrix.Matrix[2][3];

      float w = x * Matrix.Matrix[3][0] + y * Matrix.Matrix[3][1] + z * Matrix.Matrix[3][2] + Matrix.Matrix[3][3];

      if (w == 1) {
         return Point(newX, newY, newZ);
      }
      else {
         float divisor = 1.f / w;
         return Point(newX * divisor, newY * divisor, newZ * divisor);
      }
   }

   void Transform::ApplyInPlace(Vector &v) const {
      float newX = v.x * Matrix.Matrix[0][0] + v.y * Matrix.Matrix[0][1] + v.z * Matrix.Matrix[0][2];
      float newY = v.x * Matrix.Matrix[1][0] + v.y * Matrix.Matrix[1][1] + v.z * Matrix.Matrix[1][2];
      float newZ = v.x * Matrix.Matrix[2][0] + v.y * Matrix.Matrix[2][1] + v.z * Matrix.Matrix[2][2];

      v.x = newX;
      v.y = newY;
      v.z = newZ;
   }

   Vector Transform::Apply(const Vector &v) const {
      float newX = v.x * Matrix.Matrix[0][0] + v.y * Matrix.Matrix[0][1] + v.z * Matrix.Matrix[0][2];
      float newY = v.x * Matrix.Matrix[1][0] + v.y * Matrix.Matrix[1][1] + v.z * Matrix.Matrix[1][2];
      float newZ = v.x * Matrix.Matrix[2][0] + v.y * Matrix.Matrix[2][1] + v.z * Matrix.Matrix[2][2];

      Vector v1 = Vector(newX, newY, newZ);

      //v1.Normalize();

      return v1;
   }

   void Transform::ApplyInPlace(Normal &n) const {
      float newX = n.x * Inverse.Matrix[0][0] + n.y * Inverse.Matrix[1][0] + n.z * Inverse.Matrix[2][0];
      float newY = n.x * Inverse.Matrix[0][1] + n.y * Inverse.Matrix[1][1] + n.z * Inverse.Matrix[2][1];
      float newZ = n.x * Inverse.Matrix[0][2] + n.y * Inverse.Matrix[1][2] + n.z * Inverse.Matrix[2][2];

      n.x = newX;
      n.y = newY;
      n.z = newZ;
   }

   Normal Transform::Apply(const Normal &n) const {

      float newX = n.x * Inverse.Matrix[0][0] + n.y * Inverse.Matrix[1][0] + n.z * Inverse.Matrix[2][0];
      float newY = n.x * Inverse.Matrix[0][1] + n.y * Inverse.Matrix[1][1] + n.z * Inverse.Matrix[2][1];
      float newZ = n.x * Inverse.Matrix[0][2] + n.y * Inverse.Matrix[1][2] + n.z * Inverse.Matrix[2][2];

      Normal normal(newX, newY, newZ);

      normal.Normalize();

      return normal;
   }

   void Transform::ApplyInPlace(Ray &ray) const {
      ApplyInPlace(ray.Origin);
      ApplyInPlace(ray.Direction);
   }

   Ray Transform::Apply(const Ray &ray) const {
      Point p = (*this).Apply(ray.Origin);
      Vector v = (*this).Apply(ray.Direction);

      //v.Normalize();
      Ray r(p, v);

      //std::cout << "o.x: " << r.Origin.x << ", o.y: " << r.Origin.y << ", o.z: " << r.Origin.z << std::endl;
      //std::cout << "d.x: " << r.Direction.x << ", d.y: " << r.Direction.y << ", d.z: " << r.Direction.z << std::endl;

      return r;

   }

   void Transform::ApplyInPlace(BoundingBox &bb) const {
      Point p0 = Apply(Point(bb.p0.x, bb.p0.y, bb.p0.z));
      Point p1 = Apply(Point(bb.p0.x, bb.p0.y, bb.p1.z));
      Point p2 = Apply(Point(bb.p0.x, bb.p1.y, bb.p0.z));
      Point p3 = Apply(Point(bb.p0.x, bb.p1.y, bb.p1.z));
      Point p4 = Apply(Point(bb.p1.x, bb.p0.y, bb.p0.z));
      Point p5 = Apply(Point(bb.p1.x, bb.p0.y, bb.p1.z));
      Point p6 = Apply(Point(bb.p1.x, bb.p1.y, bb.p0.z));
      Point p7 = Apply(Point(bb.p1.x, bb.p1.y, bb.p1.z));

      bb.p0 = p0;
      bb.p1 = p1;

//      bb.UnionInPlace(p0);
//      bb.UnionInPlace(p1);
      bb.UnionInPlace(p2);
      bb.UnionInPlace(p3);
      bb.UnionInPlace(p4);
      bb.UnionInPlace(p5);
      bb.UnionInPlace(p6);
      bb.UnionInPlace(p7);
   }

   BoundingBox Transform::Apply(const BoundingBox &bb) const {
      return BoundingBox();
   }

   bool Transform::HasScale() const {
      float lengthX = Apply(xDir).LengthSquared();
      float lengthY = Apply(yDir).LengthSquared();
      float lengthZ = Apply(zDir).LengthSquared();

      return (lengthX < .999 || lengthX > 1.001
              || lengthY < .999 || lengthY > 1.001
              || lengthZ < .999 || lengthZ > 1.001);
   }

   Transform Transform::Rotate(const float angle, const float x, const float y, const float z) {
      const float sin = std::sin(angle);
      const float cos = std::cos(angle);
      const float oneMinusCos = (1 - cos);
      Matrix4x4 matrix = Matrix4x4(
            cos + x * x * oneMinusCos,     x * y * oneMinusCos - z * sin, x * z * oneMinusCos + y * sin, 0,
            y * x * oneMinusCos + z * sin, cos + y * y * oneMinusCos,     y * z * oneMinusCos - x * sin, 0,
            z * x * oneMinusCos - y * sin, z * y * oneMinusCos + x * sin, cos + z * z * oneMinusCos,     0,
            0,                             0,                             0,                             1
            );

      Transform transform = Transform(matrix);

      return transform;
   }

   Transform Transform::Translate(const Vector &delta) {
      Matrix4x4 matrix = Matrix4x4(
            1, 0, 0, delta.x,
            0, 1, 0, delta.y,
            0, 0, 1, delta.z,
            0, 0, 0, 1);

      Matrix4x4 inverse = Matrix4x4(
            1, 0, 0, -delta.x,
            0, 1, 0, -delta.y,
            0, 0, 1, -delta.z,
            0, 0, 0, 1);

      Transform transform = Transform(matrix, inverse);

      return transform;
   }

   Transform Transform::Translate(const float x, const float y, const float z)  {
      Matrix4x4 matrix = Matrix4x4(
            1, 0, 0, x,
            0, 1, 0, y,
            0, 0, 1, z,
            0, 0, 0, 1);

      Matrix4x4 inverse = Matrix4x4(
            1, 0, 0, -x,
            0, 1, 0, -y,
            0, 0, 1, -z,
            0, 0, 0, 1);

      Transform transform = Transform(matrix, inverse);

      return transform;
   }

   Transform Transform::Scale(const Vector &delta) {
      Matrix4x4 matrix = Matrix4x4(
            delta.x, 0, 0, 0,
            0, delta.y, 0, 0,
            0, 0, delta.z, 0,
            0, 0, 0, 1);

      Matrix4x4 inverse = Matrix4x4(
            1.f/delta.x, 0,    0,    0,
            0,    1.f/delta.y, 0,    0,
            0,    0,    1.f/delta.z, 0,
            0,    0,    0,    1);

      Transform transform = Transform(matrix, inverse);

      return transform;
   }

   Transform Transform::Scale(const float x, const float y, const float z) {
      Matrix4x4 matrix = Matrix4x4(
            x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1);

      Matrix4x4 inverse = Matrix4x4(
            1.f/x, 0,    0,    0,
            0,    1.f/y, 0,    0,
            0,    0,    1.f/z, 0,
            0,    0,    0,    1);

      Transform transform = Transform(matrix, inverse);

      return transform;
   }

   Transform Transform::Scale(const float t) {
      Matrix4x4 matrix = Matrix4x4(
            t, 0, 0, 0,
            0, t, 0, 0,
            0, 0, t, 0,
            0, 0, 0, 1);

      const float oneOverT = 1.0f / t;

      Matrix4x4 inverse = Matrix4x4(
            oneOverT, 0,    0,    0,
            0,    oneOverT, 0,    0,
            0,    0,    oneOverT, 0,
            0,    0,    0,    1);

      Transform transform = Transform(matrix, inverse);

      return transform;
   }

   Transform Transform::LookAt(const Point &eye, const Point &lookAt, Vector &up, bool right_handed) {

      float m[4][4];

      m[0][3] = eye.x;
      m[1][3] = eye.y;
      m[2][3] = eye.z;
      m[3][3] = 1;

      Vector dir = lookAt - eye;
      dir.Normalize();

      up.Normalize();
      Vector left = up.Cross(dir);

      if (left.Length() == 0) {
         std::cout << "Bad Transform::LookAt() call - left vector is 0. Up and viewing direction are pointing in the same direction. Using identity.";
         return Transform();
      }

      left.Normalize();

      const Vector newUp = dir.Cross(left);

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

      Matrix4x4 matrix = Matrix4x4(m);
      const Matrix4x4 inverse = matrix.Inverse();
      return Transform(matrix, inverse);
//      return Transform(inverse, matrix);
   }

   Transform Transform::LookAtLeftHanded(const Point &eye, const Point &lookAt, Vector &up) {

      // need to flip the z-coordinate, since polytope is right-handed internally
      const Point fixedEye(eye.x, eye.y, -eye.z);
      const Point fixedLookAt(lookAt.x, lookAt.y, lookAt.z);
      Vector fixedUp(up.x, up.y, -up.z);

      float m[4][4];

      m[0][3] = fixedEye.x;
      m[1][3] = fixedEye.y;
      m[2][3] = fixedEye.z;
      m[3][3] = 1;

      Vector dir = fixedLookAt - fixedEye;
      dir.Normalize();

      fixedUp.Normalize();
      Vector left = fixedUp.Cross(dir);

      if (left.Length() == 0) {

         std::cout << "Bad Transform::LookAt() call - left vector is 0. Up and viewing direction are pointing in the same direction. Using identity.";
         return Transform();

      }

      left.Normalize();

      const Vector newUp = dir.Cross(left);

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

      Matrix4x4 matrix = Matrix4x4(m);
      const Matrix4x4 inverse = matrix.Inverse();

      return Transform(inverse, matrix);
   }

   void Transform::ApplyPoint(float &x, float &y, float &z) const {
      float newX = x * Matrix.Matrix[0][0] + y * Matrix.Matrix[0][1] + z * Matrix.Matrix[0][2] + Matrix.Matrix[0][3];
      float newY = x * Matrix.Matrix[1][0] + y * Matrix.Matrix[1][1] + z * Matrix.Matrix[1][2] + Matrix.Matrix[1][3];
      float newZ = x * Matrix.Matrix[2][0] + y * Matrix.Matrix[2][1] + z * Matrix.Matrix[2][2] + Matrix.Matrix[2][3];

      float w = x * Matrix.Matrix[3][0] + y * Matrix.Matrix[3][1] + z * Matrix.Matrix[3][2] + Matrix.Matrix[3][3];

      x = newX;
      y = newY;
      z = newZ;
      
      if (w != 1) {
         const float divisor = 1.f / w;
         newX *= divisor;
         newY *= divisor;
         newZ *= divisor;
      }
   }
}
