//
// Created by Daniel Thompson on 2/19/18.
//

#include <cstring>
#include <exception>
#include <stdexcept>
#include "matrix.h"

namespace poly {

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

   matrix matrix::operator*(matrix rhs) const {
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

   matrix &matrix::operator*=(const matrix &rhs) {
      
      float m[16];
      
      m[0] = mat[0][0] * rhs.mat[0][0] + mat[0][1] * rhs.mat[1][0] + mat[0][2] * rhs.mat[2][0] +
                        mat[0][3] * rhs.mat[3][0];
      m[1] = mat[0][0] * rhs.mat[0][1] + mat[0][1] * rhs.mat[1][1] + mat[0][2] * rhs.mat[2][1] +
                        mat[0][3] * rhs.mat[3][1];
      m[2] = mat[0][0] * rhs.mat[0][2] + mat[0][1] * rhs.mat[1][2] + mat[0][2] * rhs.mat[2][2] +
                        mat[0][3] * rhs.mat[3][2];
      m[3] = mat[0][0] * rhs.mat[0][3] + mat[0][1] * rhs.mat[1][3] + mat[0][2] * rhs.mat[2][3] +
                        mat[0][3] * rhs.mat[3][3];

      m[4] = mat[1][0] * rhs.mat[0][0] + mat[1][1] * rhs.mat[1][0] + mat[1][2] * rhs.mat[2][0] +
                        mat[1][3] * rhs.mat[3][0];
      m[5] = mat[1][0] * rhs.mat[0][1] + mat[1][1] * rhs.mat[1][1] + mat[1][2] * rhs.mat[2][1] +
                        mat[1][3] * rhs.mat[3][1];
      m[6] = mat[1][0] * rhs.mat[0][2] + mat[1][1] * rhs.mat[1][2] + mat[1][2] * rhs.mat[2][2] +
                        mat[1][3] * rhs.mat[3][2];
      m[7] = mat[1][0] * rhs.mat[0][3] + mat[1][1] * rhs.mat[1][3] + mat[1][2] * rhs.mat[2][3] +
                        mat[1][3] * rhs.mat[3][3];

      m[8] = mat[2][0] * rhs.mat[0][0] + mat[2][1] * rhs.mat[1][0] + mat[2][2] * rhs.mat[2][0] +
                        mat[2][3] * rhs.mat[3][0];
      m[9] = mat[2][0] * rhs.mat[0][1] + mat[2][1] * rhs.mat[1][1] + mat[2][2] * rhs.mat[2][1] +
                        mat[2][3] * rhs.mat[3][1];
      m[10] = mat[2][0] * rhs.mat[0][2] + mat[2][1] * rhs.mat[1][2] + mat[2][2] * rhs.mat[2][2] +
                        mat[2][3] * rhs.mat[3][2];
      m[11] = mat[2][0] * rhs.mat[0][3] + mat[2][1] * rhs.mat[1][3] + mat[2][2] * rhs.mat[2][3] +
                        mat[2][3] * rhs.mat[3][3];

      m[12] = mat[3][0] * rhs.mat[0][0] + mat[3][1] * rhs.mat[1][0] + mat[3][2] * rhs.mat[2][0] +
                        mat[3][3] * rhs.mat[3][0];
      m[13] = mat[3][0] * rhs.mat[0][1] + mat[3][1] * rhs.mat[1][1] + mat[3][2] * rhs.mat[2][1] +
                        mat[3][3] * rhs.mat[3][1];
      m[14] = mat[3][0] * rhs.mat[0][2] + mat[3][1] * rhs.mat[1][2] + mat[3][2] * rhs.mat[2][2] +
                        mat[3][3] * rhs.mat[3][2];
      m[15] = mat[3][0] * rhs.mat[0][3] + mat[3][1] * rhs.mat[1][3] + mat[3][2] * rhs.mat[2][3] +
                        mat[3][3] * rhs.mat[3][3];

      std::memcpy(mat, m, sizeof(float) * 16);
      
      return *this;
   }
}
