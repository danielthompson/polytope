//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "gtest/gtest.h"

#include "../src/cpu/structures/matrix.h"

namespace Tests {

   using poly::matrix;

   namespace Equality {
      TEST(Matrix4x4, Equals) {
         matrix m1 = matrix();
         matrix m2 = matrix();
         EXPECT_EQ(m1, m2);
      }
   }

   namespace Multiply {
      TEST(Matrix4x4, MultiplyIdentities) {
         matrix m1 = matrix();
         matrix m2 = matrix();

         matrix actual = m1 * m2;
         matrix expected = matrix();


         EXPECT_EQ(actual, expected);
      }

      TEST(Matrix4x4, Multiply1) {
         matrix m1 = matrix(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
         matrix m2 = matrix(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

         matrix actual = m1 * m2;
         matrix expected = matrix(90, 100, 110, 120, 202, 228, 254, 280, 314, 356, 398, 440, 426, 484, 542, 600);
         EXPECT_EQ(actual, expected);
      }

      TEST(Matrix4x4, Multiply2) {
         matrix m1 = matrix(5, 3, 1, 4, 3, 4, 2, 3, 1, 2, 4, 7, 9, 5, 8, 3);
         matrix m2 = matrix(4, 6, 7, 1, 2, 3, 4, 7, 9, 6, 5, 8, 7, 4, 1, 3);

         matrix actual = m1 * m2;
         matrix expected = matrix(63, 61, 56, 46, 59, 54, 50, 56, 93, 64, 42, 68, 139, 129, 126, 117);
         EXPECT_EQ(actual, expected);
      }
   }

   namespace Inverse {
      TEST(Matrix4x4, InverseIdentity) {
         matrix m1 = matrix();

         matrix actual = m1.inverse();
         matrix expected = matrix();

         EXPECT_EQ(actual, expected);
      }
   }

   namespace Transpose {
      TEST(Matrix4x4, TransposeIdentity) {
         matrix m1 = matrix();

         matrix actual = m1.transpose();
         matrix expected = matrix();

         EXPECT_EQ(actual, expected);
      }

      TEST(Matrix4x4, Transpose1) {
         matrix m1 = matrix(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

         matrix actual = m1.transpose();

         EXPECT_EQ(actual.mat[0][0], m1.mat[0][0]);
         EXPECT_EQ(actual.mat[0][1], m1.mat[1][0]);
         EXPECT_EQ(actual.mat[0][2], m1.mat[2][0]);
         EXPECT_EQ(actual.mat[0][3], m1.mat[3][0]);
         EXPECT_EQ(actual.mat[1][0], m1.mat[0][1]);
         EXPECT_EQ(actual.mat[1][1], m1.mat[1][1]);
         EXPECT_EQ(actual.mat[1][2], m1.mat[2][1]);
         EXPECT_EQ(actual.mat[1][3], m1.mat[3][1]);
         EXPECT_EQ(actual.mat[2][0], m1.mat[0][2]);
         EXPECT_EQ(actual.mat[2][1], m1.mat[1][2]);
         EXPECT_EQ(actual.mat[2][2], m1.mat[2][2]);
         EXPECT_EQ(actual.mat[2][3], m1.mat[3][2]);
         EXPECT_EQ(actual.mat[3][0], m1.mat[0][3]);
         EXPECT_EQ(actual.mat[3][1], m1.mat[1][3]);
         EXPECT_EQ(actual.mat[3][2], m1.mat[2][3]);
         EXPECT_EQ(actual.mat[3][3], m1.mat[3][3]);
      }

      TEST(Matrix4x4, Transpose2) {
         matrix m1 = matrix(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

         matrix transposed = m1.transpose();

         matrix retransposed = transposed.transpose();

         EXPECT_EQ(m1, retransposed);
      }
   }
}