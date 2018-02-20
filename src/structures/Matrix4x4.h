//
// Created by Daniel Thompson on 2/19/18.
//

#ifndef POLYTOPE_MATRIX4X4_H
#define POLYTOPE_MATRIX4X4_H

namespace Polytope {
   class Matrix4x4 {
   public:

      float Matrix[4][4];

      Matrix4x4();

      explicit Matrix4x4(const float m[4][4]);

      Matrix4x4(float t00, float t01, float t02, float t03,
                float t10, float t11, float t12, float t13,
                float t20, float t21, float t22, float t23,
                float t30, float t31, float t32, float t33);

      Matrix4x4 Transpose();

      Matrix4x4 Inverse();

      Matrix4x4 Multiply(Matrix4x4 m);

      bool operator==(const Matrix4x4 &rhs) const;

      bool operator!=(const Matrix4x4 &rhs) const;
   };

}
#endif //POLYTOPE_MATRIX4X4_H
