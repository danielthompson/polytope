//
// Created by Daniel Thompson on 2/19/18.
//

#include <cstring>
#include <exception>
#include <stdexcept>
#include "matrix.h"

namespace poly {

   matrix::matrix() : mat {0} {
      mat[0][0] = 1;
      //Matrix[0][1] = 0;
      //Matrix[0][2] = 0;
      //Matrix[0][3] = 0;

      //Matrix[1][0] = 0;
      mat[1][1] = 1;
      //Matrix[1][2] = 0;
      //Matrix[1][3] = 0;

      //Matrix[2][0] = 0;
      //Matrix[2][1] = 0;
      mat[2][2] = 1;
      //Matrix[2][3] = 0;

      //Matrix[3][0] = 0;
      //Matrix[3][1] = 0;
      //Matrix[3][2] = 0;
      mat[3][3] = 1;
   }

   matrix::matrix(const float m[4][4]) {
      mat[0][0] = m[0][0];
      mat[0][1] = m[0][1];
      mat[0][2] = m[0][2];
      mat[0][3] = m[0][3];

      mat[1][0] = m[1][0];
      mat[1][1] = m[1][1];
      mat[1][2] = m[1][2];
      mat[1][3] = m[1][3];

      mat[2][0] = m[2][0];
      mat[2][1] = m[2][1];
      mat[2][2] = m[2][2];
      mat[2][3] = m[2][3];

      mat[3][0] = m[3][0];
      mat[3][1] = m[3][1];
      mat[3][2] = m[3][2];
      mat[3][3] = m[3][3];
   }

   matrix::matrix(float t00, float t01, float t02, float t03,
                  float t10, float t11, float t12, float t13,
                  float t20, float t21, float t22, float t23,
                  float t30, float t31, float t32, float t33) {

      mat[0][0] = t00;
      mat[0][1] = t01;
      mat[0][2] = t02;
      mat[0][3] = t03;

      mat[1][0] = t10;
      mat[1][1] = t11;
      mat[1][2] = t12;
      mat[1][3] = t13;

      mat[2][0] = t20;
      mat[2][1] = t21;
      mat[2][2] = t22;
      mat[2][3] = t23;

      mat[3][0] = t30;
      mat[3][1] = t31;
      mat[3][2] = t32;
      mat[3][3] = t33;
   }

   matrix::matrix(const matrix &other) {
      mat[0][0] = other.mat[0][0];
      mat[0][1] = other.mat[0][1];
      mat[0][2] = other.mat[0][2];
      mat[0][3] = other.mat[0][3];

      mat[1][0] = other.mat[1][0];
      mat[1][1] = other.mat[1][1];
      mat[1][2] = other.mat[1][2];
      mat[1][3] = other.mat[1][3];

      mat[2][0] = other.mat[2][0];
      mat[2][1] = other.mat[2][1];
      mat[2][2] = other.mat[2][2];
      mat[2][3] = other.mat[2][3];

      mat[3][0] = other.mat[3][0];
      mat[3][1] = other.mat[3][1];
      mat[3][2] = other.mat[3][2];
      mat[3][3] = other.mat[3][3];
   }

   matrix matrix::transpose() {
      return matrix(
            mat[0][0], mat[1][0], mat[2][0], mat[3][0],
            mat[0][1], mat[1][1], mat[2][1], mat[3][1],
            mat[0][2], mat[1][2], mat[2][2], mat[3][2],
            mat[0][3], mat[1][3], mat[2][3], mat[3][3]);
   }

   matrix matrix::inverse() {
      int indxc[4] = {0, 0, 0, 0};
      int indxr[4] = {0, 0, 0, 0};
      int ipiv[4] = {0, 0, 0, 0};
      float minv[4][4];
      memcpy(minv, mat, 4 * 4 * sizeof(float));
      for (int i = 0; i < 4; i++) {
         int irow = -1, icol = -1;
         float big = 0.f;
         // Choose pivot
         for (int j = 0; j < 4; j++) {
            if (ipiv[j] != 1) {
               for (int k = 0; k < 4; k++) {
                  if (ipiv[k] == 0) {
                     if (abs(minv[j][k]) >= big) {
                        big = abs(minv[j][k]);
                        irow = j;
                        icol = k;
                     }
                  } else if (ipiv[k] > 1)
                     throw std::logic_error("Singular matrix in inverse()");
               }
            }
         }
         ++ipiv[icol];

         // Swap rows _irow_ and _icol_ for pivot
         if (irow != icol) {
            for (int k = 0; k < 4; ++k) std::swap(minv[irow][k], minv[icol][k]);
         }
         indxr[i] = irow;
         indxc[i] = icol;
         if (minv[icol][icol] == 0.)
            throw std::logic_error("Singular matrix in inverse()");
         // Set $m[icol][icol]$ to one by scaling row _icol_ appropriately
         float pivinv = 1.f / minv[icol][icol];
         minv[icol][icol] = 1.f;
         for (int j = 0; j < 4; j++)
            minv[icol][j] *= pivinv;
         // Subtract this row from others to zero out their columns
         for (int j = 0; j < 4; j++) {
            if (j != icol) {
               float save = minv[j][icol];
               minv[j][icol] = 0;
               for (int k = 0; k < 4; k++)
                  minv[j][k] -= minv[icol][k] * save;
            }
         }
      }
      // Swap columns to reflect permutation
      for (int j = 3; j >= 0; j--) {
         if (indxr[j] != indxc[j]) {
            for (int k = 0; k < 4; k++)
               std::swap(minv[k][indxr[j]], minv[k][indxc[j]]);
         }
      }

      return matrix(minv);
   }


   matrix matrix::operator*(const matrix &rhs) const {
      matrix ret = matrix();

      ret.mat[0][0] = mat[0][0] * rhs.mat[0][0] + mat[0][1] * rhs.mat[1][0] + mat[0][2] * rhs.mat[2][0] +
                      mat[0][3] * rhs.mat[3][0];
      ret.mat[0][1] = mat[0][0] * rhs.mat[0][1] + mat[0][1] * rhs.mat[1][1] + mat[0][2] * rhs.mat[2][1] +
                      mat[0][3] * rhs.mat[3][1];
      ret.mat[0][2] = mat[0][0] * rhs.mat[0][2] + mat[0][1] * rhs.mat[1][2] + mat[0][2] * rhs.mat[2][2] +
                      mat[0][3] * rhs.mat[3][2];
      ret.mat[0][3] = mat[0][0] * rhs.mat[0][3] + mat[0][1] * rhs.mat[1][3] + mat[0][2] * rhs.mat[2][3] +
                      mat[0][3] * rhs.mat[3][3];

      ret.mat[1][0] = mat[1][0] * rhs.mat[0][0] + mat[1][1] * rhs.mat[1][0] + mat[1][2] * rhs.mat[2][0] +
                      mat[1][3] * rhs.mat[3][0];
      ret.mat[1][1] = mat[1][0] * rhs.mat[0][1] + mat[1][1] * rhs.mat[1][1] + mat[1][2] * rhs.mat[2][1] +
                      mat[1][3] * rhs.mat[3][1];
      ret.mat[1][2] = mat[1][0] * rhs.mat[0][2] + mat[1][1] * rhs.mat[1][2] + mat[1][2] * rhs.mat[2][2] +
                      mat[1][3] * rhs.mat[3][2];
      ret.mat[1][3] = mat[1][0] * rhs.mat[0][3] + mat[1][1] * rhs.mat[1][3] + mat[1][2] * rhs.mat[2][3] +
                      mat[1][3] * rhs.mat[3][3];

      ret.mat[2][0] = mat[2][0] * rhs.mat[0][0] + mat[2][1] * rhs.mat[1][0] + mat[2][2] * rhs.mat[2][0] +
                      mat[2][3] * rhs.mat[3][0];
      ret.mat[2][1] = mat[2][0] * rhs.mat[0][1] + mat[2][1] * rhs.mat[1][1] + mat[2][2] * rhs.mat[2][1] +
                      mat[2][3] * rhs.mat[3][1];
      ret.mat[2][2] = mat[2][0] * rhs.mat[0][2] + mat[2][1] * rhs.mat[1][2] + mat[2][2] * rhs.mat[2][2] +
                      mat[2][3] * rhs.mat[3][2];
      ret.mat[2][3] = mat[2][0] * rhs.mat[0][3] + mat[2][1] * rhs.mat[1][3] + mat[2][2] * rhs.mat[2][3] +
                      mat[2][3] * rhs.mat[3][3];

      ret.mat[3][0] = mat[3][0] * rhs.mat[0][0] + mat[3][1] * rhs.mat[1][0] + mat[3][2] * rhs.mat[2][0] +
                      mat[3][3] * rhs.mat[3][0];
      ret.mat[3][1] = mat[3][0] * rhs.mat[0][1] + mat[3][1] * rhs.mat[1][1] + mat[3][2] * rhs.mat[2][1] +
                      mat[3][3] * rhs.mat[3][1];
      ret.mat[3][2] = mat[3][0] * rhs.mat[0][2] + mat[3][1] * rhs.mat[1][2] + mat[3][2] * rhs.mat[2][2] +
                      mat[3][3] * rhs.mat[3][2];
      ret.mat[3][3] = mat[3][0] * rhs.mat[0][3] + mat[3][1] * rhs.mat[1][3] + mat[3][2] * rhs.mat[2][3] +
                      mat[3][3] * rhs.mat[3][3];

      return ret;
   }

   bool matrix::operator==(const matrix &rhs) const {

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
   }

   bool matrix::operator!=(const matrix &rhs) const {
      return !(rhs == *this);
   }

   matrix &matrix::operator*=(const matrix &rhs) {
      
      const float m00 = mat[0][0] * rhs.mat[0][0] + mat[0][1] * rhs.mat[1][0] + mat[0][2] * rhs.mat[2][0] +
                        mat[0][3] * rhs.mat[3][0];
      const float m01 = mat[0][0] * rhs.mat[0][1] + mat[0][1] * rhs.mat[1][1] + mat[0][2] * rhs.mat[2][1] +
                        mat[0][3] * rhs.mat[3][1];
      const float m02 = mat[0][0] * rhs.mat[0][2] + mat[0][1] * rhs.mat[1][2] + mat[0][2] * rhs.mat[2][2] +
                        mat[0][3] * rhs.mat[3][2];
      const float m03 = mat[0][0] * rhs.mat[0][3] + mat[0][1] * rhs.mat[1][3] + mat[0][2] * rhs.mat[2][3] +
                        mat[0][3] * rhs.mat[3][3];

      const float m10 = mat[1][0] * rhs.mat[0][0] + mat[1][1] * rhs.mat[1][0] + mat[1][2] * rhs.mat[2][0] +
                        mat[1][3] * rhs.mat[3][0];
      const float m11 = mat[1][0] * rhs.mat[0][1] + mat[1][1] * rhs.mat[1][1] + mat[1][2] * rhs.mat[2][1] +
                        mat[1][3] * rhs.mat[3][1];
      const float m12 = mat[1][0] * rhs.mat[0][2] + mat[1][1] * rhs.mat[1][2] + mat[1][2] * rhs.mat[2][2] +
                        mat[1][3] * rhs.mat[3][2];
      const float m13 = mat[1][0] * rhs.mat[0][3] + mat[1][1] * rhs.mat[1][3] + mat[1][2] * rhs.mat[2][3] +
                        mat[1][3] * rhs.mat[3][3];

      const float m20 = mat[2][0] * rhs.mat[0][0] + mat[2][1] * rhs.mat[1][0] + mat[2][2] * rhs.mat[2][0] +
                        mat[2][3] * rhs.mat[3][0];
      const float m21 = mat[2][0] * rhs.mat[0][1] + mat[2][1] * rhs.mat[1][1] + mat[2][2] * rhs.mat[2][1] +
                        mat[2][3] * rhs.mat[3][1];
      const float m22 = mat[2][0] * rhs.mat[0][2] + mat[2][1] * rhs.mat[1][2] + mat[2][2] * rhs.mat[2][2] +
                        mat[2][3] * rhs.mat[3][2];
      const float m23 = mat[2][0] * rhs.mat[0][3] + mat[2][1] * rhs.mat[1][3] + mat[2][2] * rhs.mat[2][3] +
                        mat[2][3] * rhs.mat[3][3];

      const float m30 = mat[3][0] * rhs.mat[0][0] + mat[3][1] * rhs.mat[1][0] + mat[3][2] * rhs.mat[2][0] +
                        mat[3][3] * rhs.mat[3][0];
      const float m31 = mat[3][0] * rhs.mat[0][1] + mat[3][1] * rhs.mat[1][1] + mat[3][2] * rhs.mat[2][1] +
                        mat[3][3] * rhs.mat[3][1];
      const float m32 = mat[3][0] * rhs.mat[0][2] + mat[3][1] * rhs.mat[1][2] + mat[3][2] * rhs.mat[2][2] +
                        mat[3][3] * rhs.mat[3][2];
      const float m33 = mat[3][0] * rhs.mat[0][3] + mat[3][1] * rhs.mat[1][3] + mat[3][2] * rhs.mat[2][3] +
                        mat[3][3] * rhs.mat[3][3];

      mat[0][0] = m00;
      mat[0][1] = m01;
      mat[0][2] = m02;
      mat[0][3] = m03;

      mat[1][0] = m10;
      mat[1][1] = m11;
      mat[1][2] = m12;
      mat[1][3] = m13;

      mat[2][0] = m20;
      mat[2][1] = m21;
      mat[2][2] = m22;
      mat[2][3] = m23;

      mat[3][0] = m30;
      mat[3][1] = m31;
      mat[3][2] = m32;
      mat[3][3] = m33;
      
      return *this;
   }
}
