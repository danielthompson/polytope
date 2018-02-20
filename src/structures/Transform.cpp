//
// Created by Daniel Thompson on 2/19/18.
//

#include "Transform.h"

namespace Polytope::Structures {

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

   Transform Transform::Invert() {
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


}