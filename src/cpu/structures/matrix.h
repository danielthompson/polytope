//
// Created by Daniel Thompson on 2/19/18.
//

#ifndef POLY_MATRIX4X4_H
#define POLY_MATRIX4X4_H

namespace poly {
   class matrix {
   public:
      matrix();
      explicit matrix(const float m[4][4]);
      matrix(float t00, float t01, float t02, float t03,
             float t10, float t11, float t12, float t13,
             float t20, float t21, float t22, float t23,
             float t30, float t31, float t32, float t33);

      matrix(const matrix &other);

      bool operator==(const matrix &rhs) const;
      bool operator!=(const matrix &rhs) const;
      matrix operator*(const matrix &rhs) const;
      matrix &operator*=(const matrix &rhs);

      matrix transpose();
      matrix inverse();

      float mat[4][4]{};
   };
}
#endif //POLY_MATRIX4X4_H
