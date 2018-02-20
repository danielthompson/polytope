//
// Created by Daniel Thompson on 2/19/18.
//

#include <cstring>
#include <cmath>
#include <exception>
#include <stdexcept>
#include "Matrix4x4.h"

namespace Polytope::Structures {

   Matrix4x4::Matrix4x4() {
      Matrix[0][0] = 1;
      Matrix[0][1] = 0;
      Matrix[0][2] = 0;
      Matrix[0][3] = 0;

      Matrix[1][0] = 0;
      Matrix[1][1] = 1;
      Matrix[1][2] = 0;
      Matrix[1][3] = 0;

      Matrix[2][0] = 0;
      Matrix[2][1] = 0;
      Matrix[2][2] = 1;
      Matrix[2][3] = 0;

      Matrix[3][0] = 0;
      Matrix[3][1] = 0;
      Matrix[3][2] = 0;
      Matrix[3][3] = 1;
   }

   Matrix4x4::Matrix4x4(const float m[4][4]) {
      Matrix[0][0] = m[0][0];
      Matrix[0][1] = m[0][1];
      Matrix[0][2] = m[0][2];
      Matrix[0][3] = m[0][3];

      Matrix[1][0] = m[1][0];
      Matrix[1][1] = m[1][1];
      Matrix[1][2] = m[1][2];
      Matrix[1][3] = m[1][3];

      Matrix[2][0] = m[2][0];
      Matrix[2][1] = m[2][1];
      Matrix[2][2] = m[2][2];
      Matrix[2][3] = m[2][3];

      Matrix[3][0] = m[3][0];
      Matrix[3][1] = m[3][1];
      Matrix[3][2] = m[3][2];
      Matrix[3][3] = m[3][3];
   }

   Matrix4x4::Matrix4x4(float t00, float t01, float t02, float t03,
                        float t10, float t11, float t12, float t13,
                        float t20, float t21, float t22, float t23,
                        float t30, float t31, float t32, float t33) {

      Matrix[0][0] = t00;
      Matrix[0][1] = t01;
      Matrix[0][2] = t02;
      Matrix[0][3] = t03;

      Matrix[1][0] = t10;
      Matrix[1][1] = t11;
      Matrix[1][2] = t12;
      Matrix[1][3] = t13;

      Matrix[2][0] = t20;
      Matrix[2][1] = t21;
      Matrix[2][2] = t22;
      Matrix[2][3] = t23;

      Matrix[3][0] = t30;
      Matrix[3][1] = t31;
      Matrix[3][2] = t32;
      Matrix[3][3] = t33;
   }

   Matrix4x4 Matrix4x4::Transpose() {
      return Matrix4x4(
            Matrix[0][0], Matrix[1][0], Matrix[2][0], Matrix[3][0],
            Matrix[0][1], Matrix[1][1], Matrix[2][1], Matrix[3][1],
            Matrix[0][2], Matrix[1][2], Matrix[2][2], Matrix[3][2],
            Matrix[0][3], Matrix[1][3], Matrix[2][3], Matrix[3][3]);
   }

   Matrix4x4 Matrix4x4::Inverse() {
      int indxc[4] = {0, 0, 0, 0};
      int indxr[4] = {0, 0, 0, 0};
      int ipiv[4] = {0, 0, 0, 0};
      float minv[4][4];
      memcpy(minv, Matrix, 4 * 4 * sizeof(float));
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
                     throw std::logic_error("Singular matrix in MatrixInvert");
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
            throw std::logic_error("Singular matrix in MatrixInvert");
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

      return Matrix4x4(minv);
   }


   Matrix4x4 Matrix4x4::Multiply(Matrix4x4 m) {
      Matrix4x4 ret = Matrix4x4();

      ret.Matrix[0][0] = Matrix[0][0] * m.Matrix[0][0] + Matrix[0][1] * m.Matrix[1][0] + Matrix[0][2] * m.Matrix[2][0] +
                         Matrix[0][3] * m.Matrix[3][0];
      ret.Matrix[0][1] = Matrix[0][0] * m.Matrix[0][1] + Matrix[0][1] * m.Matrix[1][1] + Matrix[0][2] * m.Matrix[2][1] +
                         Matrix[0][3] * m.Matrix[3][1];
      ret.Matrix[0][2] = Matrix[0][0] * m.Matrix[0][2] + Matrix[0][1] * m.Matrix[1][2] + Matrix[0][2] * m.Matrix[2][2] +
                         Matrix[0][3] * m.Matrix[3][2];
      ret.Matrix[0][3] = Matrix[0][0] * m.Matrix[0][3] + Matrix[0][1] * m.Matrix[1][3] + Matrix[0][2] * m.Matrix[2][3] +
                         Matrix[0][3] * m.Matrix[3][3];

      ret.Matrix[1][0] = Matrix[1][0] * m.Matrix[0][0] + Matrix[1][1] * m.Matrix[1][0] + Matrix[1][2] * m.Matrix[2][0] +
                         Matrix[1][3] * m.Matrix[3][0];
      ret.Matrix[1][1] = Matrix[1][0] * m.Matrix[0][1] + Matrix[1][1] * m.Matrix[1][1] + Matrix[1][2] * m.Matrix[2][1] +
                         Matrix[1][3] * m.Matrix[3][1];
      ret.Matrix[1][2] = Matrix[1][0] * m.Matrix[0][2] + Matrix[1][1] * m.Matrix[1][2] + Matrix[1][2] * m.Matrix[2][2] +
                         Matrix[1][3] * m.Matrix[3][2];
      ret.Matrix[1][3] = Matrix[1][0] * m.Matrix[0][3] + Matrix[1][1] * m.Matrix[1][3] + Matrix[1][2] * m.Matrix[2][3] +
                         Matrix[1][3] * m.Matrix[3][3];

      ret.Matrix[2][0] = Matrix[2][0] * m.Matrix[0][0] + Matrix[2][1] * m.Matrix[1][0] + Matrix[2][2] * m.Matrix[2][0] +
                         Matrix[2][3] * m.Matrix[3][0];
      ret.Matrix[2][1] = Matrix[2][0] * m.Matrix[0][1] + Matrix[2][1] * m.Matrix[1][1] + Matrix[2][2] * m.Matrix[2][1] +
                         Matrix[2][3] * m.Matrix[3][1];
      ret.Matrix[2][2] = Matrix[2][0] * m.Matrix[0][2] + Matrix[2][1] * m.Matrix[1][2] + Matrix[2][2] * m.Matrix[2][2] +
                         Matrix[2][3] * m.Matrix[3][2];
      ret.Matrix[2][3] = Matrix[2][0] * m.Matrix[0][3] + Matrix[2][1] * m.Matrix[1][3] + Matrix[2][2] * m.Matrix[2][3] +
                         Matrix[2][3] * m.Matrix[3][3];

      ret.Matrix[3][0] = Matrix[3][0] * m.Matrix[0][0] + Matrix[3][1] * m.Matrix[1][0] + Matrix[3][2] * m.Matrix[2][0] +
                         Matrix[3][3] * m.Matrix[3][0];
      ret.Matrix[3][1] = Matrix[3][0] * m.Matrix[0][1] + Matrix[3][1] * m.Matrix[1][1] + Matrix[3][2] * m.Matrix[2][1] +
                         Matrix[3][3] * m.Matrix[3][1];
      ret.Matrix[3][2] = Matrix[3][0] * m.Matrix[0][2] + Matrix[3][1] * m.Matrix[1][2] + Matrix[3][2] * m.Matrix[2][2] +
                         Matrix[3][3] * m.Matrix[3][2];
      ret.Matrix[3][3] = Matrix[3][0] * m.Matrix[0][3] + Matrix[3][1] * m.Matrix[1][3] + Matrix[3][2] * m.Matrix[2][3] +
                         Matrix[3][3] * m.Matrix[3][3];

      return ret;
   }

   bool Matrix4x4::operator==(const Matrix4x4 &rhs) const {

      return (
            Matrix[0][0] == rhs.Matrix[0][0] &&
            Matrix[0][1] == rhs.Matrix[0][1] &&
            Matrix[0][2] == rhs.Matrix[0][2] &&
            Matrix[0][3] == rhs.Matrix[0][3] &&

            Matrix[1][0] == rhs.Matrix[1][0] &&
            Matrix[1][1] == rhs.Matrix[1][1] &&
            Matrix[1][2] == rhs.Matrix[1][2] &&
            Matrix[1][3] == rhs.Matrix[1][3] &&

            Matrix[2][0] == rhs.Matrix[2][0] &&
            Matrix[2][1] == rhs.Matrix[2][1] &&
            Matrix[2][2] == rhs.Matrix[2][2] &&
            Matrix[2][3] == rhs.Matrix[2][3] &&

            Matrix[3][0] == rhs.Matrix[3][0] &&
            Matrix[3][1] == rhs.Matrix[3][1] &&
            Matrix[3][2] == rhs.Matrix[3][2] &&
            Matrix[3][3] == rhs.Matrix[3][3]
      );
   }

   bool Matrix4x4::operator!=(const Matrix4x4 &rhs) const {
      return !(rhs == *this);
   }

}