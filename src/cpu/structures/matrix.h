//
// Created by Daniel Thompson on 2/19/18.
//

#ifndef POLY_MATRIX4X4_H
#define POLY_MATRIX4X4_H

namespace poly {
   class matrix {
   public:
      constexpr matrix() : mat { 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f } { }
      constexpr explicit matrix(const float m[4][4]) : mat {
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3]
      } { }
      constexpr matrix(float t00, float t01, float t02, float t03,
                       float t10, float t11, float t12, float t13,
                       float t20, float t21, float t22, float t23,
                       float t30, float t31, float t32, float t33) : mat {
            t00, t01, t02, t03,
            t10, t11, t12, t13,
            t20, t21, t22, t23,
            t30, t31, t32, t33
      } { }

      constexpr matrix(const matrix &other) = default;

      constexpr bool operator==(const matrix &rhs) const {
         return (
               mat[0][0] == rhs.mat[0][0] &&
               mat[0][1] == rhs.mat[0][1] &&
               mat[0][2] == rhs.mat[0][2] &&
               mat[0][3] == rhs.mat[0][3] &&

               mat[1][0] == rhs.mat[1][0] &&
               mat[1][1] == rhs.mat[1][1] &&
               mat[1][2] == rhs.mat[1][2] &&
               mat[1][3] == rhs.mat[1][3] &&

               mat[2][0] == rhs.mat[2][0] &&
               mat[2][1] == rhs.mat[2][1] &&
               mat[2][2] == rhs.mat[2][2] &&
               mat[2][3] == rhs.mat[2][3] &&

               mat[3][0] == rhs.mat[3][0] &&
               mat[3][1] == rhs.mat[3][1] &&
               mat[3][2] == rhs.mat[3][2] &&
               mat[3][3] == rhs.mat[3][3]
         );
      };

      constexpr bool operator!=(const matrix &rhs) const {
         return !(rhs == *this);
      }
      
      matrix operator*(const matrix &rhs) const;
      matrix &operator*=(const matrix &rhs);

      constexpr matrix transpose() {
         return {
               mat[0][0], mat[1][0], mat[2][0], mat[3][0],
               mat[0][1], mat[1][1], mat[2][1], mat[3][1],
               mat[0][2], mat[1][2], mat[2][2], mat[3][2],
               mat[0][3], mat[1][3], mat[2][3], mat[3][3]
         };
      }
      matrix inverse();

      float mat[4][4];
   };
}
#endif //POLY_MATRIX4X4_H
