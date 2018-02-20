//
// Created by Daniel Thompson on 2/19/18.
//

#include "Transform.h"

namespace Polytope {

   Transform::Transform() {
      Matrix = Matrix4x4();
      Inverse = Matrix4x4();
   }

   Transform::Transform(const float values[4][4]) {
      Matrix = Matrix4x4(values);
      Inverse = Matrix.Inverse();
   }

   Transform::Transform(const Matrix4x4 matrix) {
      Matrix = matrix;
      Inverse = Matrix.Inverse();
   }

   Transform::Transform(const Matrix4x4 matrix, const Matrix4x4 inverse) {
      Matrix = matrix;
      Inverse = inverse;
   }

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

   void Transform::ApplyInPlace(Point &p) const {
      float x = p.x;
      float y = p.y;
      float z = p.z;

      float newX = x * Matrix.Matrix[0][0] + y * Matrix.Matrix[0][1] + z * Matrix.Matrix[0][2] + Matrix.Matrix[0][3];
      float newY = x * Matrix.Matrix[1][0] + y * Matrix.Matrix[1][1] + z * Matrix.Matrix[1][2] + Matrix.Matrix[1][3];
      float newZ = x * Matrix.Matrix[2][0] + y * Matrix.Matrix[2][1] + z * Matrix.Matrix[2][2] + Matrix.Matrix[2][3];

      float w = x * Matrix.Matrix[3][0] + y * Matrix.Matrix[3][1] + z * Matrix.Matrix[3][2] + Matrix.Matrix[3][3];

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

      return Vector(newX, newY, newZ);
   }

   void Transform::ApplyInPlace(Ray &ray) const {
      ApplyInPlace(ray.Origin);
      ApplyInPlace(ray.Direction);
      ApplyInPlace(ray.DirectionInverse);
   }

   Ray Transform::Apply(const Ray &ray) const {
      return Ray(Apply(ray.Origin), Apply(ray.Direction));

   }

}