//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "gtest/gtest.h"

#include "../src/structures/Matrix4x4.h"

namespace Tests::Matrix4x4 {

   using Polytope::Structures::Matrix4x4;

   namespace Equality {
      TEST(Matrix4x4, Equals) {
         Matrix4x4 m1 = Matrix4x4();
         Matrix4x4 m2 = Matrix4x4();
         EXPECT_EQ(m1, m2);
      }
   }

   namespace Multiply {
      TEST(Matrix4x4, MultiplyIdentities) {
         Matrix4x4 m1 = Matrix4x4();
         Matrix4x4 m2 = Matrix4x4();

         Matrix4x4 actual = m1.Multiply(m2);
         Matrix4x4 expected = Matrix4x4();


         EXPECT_EQ(actual, expected);
      }

      TEST(Matrix4x4, Multiply1) {
         Matrix4x4 m1 = Matrix4x4( 1,   2,   3,   4,    5,   6,   7,   8,    9,  10,  11,  12,   13,  14,  15,  16);
         Matrix4x4 m2 = Matrix4x4( 1,   2,   3,   4,    5,   6,   7,   8,    9,  10,  11,  12,   13,  14,  15,  16);

         Matrix4x4 actual = m1.Multiply(m2);
         Matrix4x4 expected = Matrix4x4(90, 100, 110, 120,  202, 228, 254, 280,  314, 356, 398, 440,  426, 484, 542, 600);
         EXPECT_EQ(actual, expected);
      }

      TEST(Matrix4x4, Multiply2) {
         Matrix4x4 m1 = Matrix4x4( 5,  3,  1,  4,   3,  4,  2,  3,   1,  2,  4,  7,    9,   5,   8,   3);
         Matrix4x4 m2 = Matrix4x4( 4,  6,  7,  1,   2,  3,  4,  7,   9,  6,  5,  8,    7,   4,   1,   3);

         Matrix4x4 actual = m1.Multiply(m2);
         Matrix4x4 expected = Matrix4x4(63, 61, 56, 46,  59, 54, 50, 56,  93, 64, 42, 68,  139, 129, 126, 117);
         EXPECT_EQ(actual, expected);
      }
   }

   namespace Inverse {
      TEST(Matrix4x4, InverseIdentity) {
         Matrix4x4 m1 = Matrix4x4();

         Matrix4x4 actual = m1.Inverse();
         Matrix4x4 expected = Matrix4x4();

         EXPECT_EQ(actual, expected);
      }
   }

   namespace Transpose {
      TEST(Matrix4x4, TransposeIdentity) {
         Matrix4x4 m1 = Matrix4x4();

         Matrix4x4 actual = m1.Transpose();
         Matrix4x4 expected = Matrix4x4();

         EXPECT_EQ(actual, expected);
      }

      TEST(Matrix4x4, Transpose1) {
         Matrix4x4 m1 = Matrix4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

         Matrix4x4 actual = m1.Transpose();

         EXPECT_EQ(actual.Matrix[0][0], m1.Matrix[0][0]);
         EXPECT_EQ(actual.Matrix[0][1], m1.Matrix[1][0]);
         EXPECT_EQ(actual.Matrix[0][2], m1.Matrix[2][0]);
         EXPECT_EQ(actual.Matrix[0][3], m1.Matrix[3][0]);
         EXPECT_EQ(actual.Matrix[1][0], m1.Matrix[0][1]);
         EXPECT_EQ(actual.Matrix[1][1], m1.Matrix[1][1]);
         EXPECT_EQ(actual.Matrix[1][2], m1.Matrix[2][1]);
         EXPECT_EQ(actual.Matrix[1][3], m1.Matrix[3][1]);
         EXPECT_EQ(actual.Matrix[2][0], m1.Matrix[0][2]);
         EXPECT_EQ(actual.Matrix[2][1], m1.Matrix[1][2]);
         EXPECT_EQ(actual.Matrix[2][2], m1.Matrix[2][2]);
         EXPECT_EQ(actual.Matrix[2][3], m1.Matrix[3][2]);
         EXPECT_EQ(actual.Matrix[3][0], m1.Matrix[0][3]);
         EXPECT_EQ(actual.Matrix[3][1], m1.Matrix[1][3]);
         EXPECT_EQ(actual.Matrix[3][2], m1.Matrix[2][3]);
         EXPECT_EQ(actual.Matrix[3][3], m1.Matrix[3][3]);
      }

      TEST(Matrix4x4, Transpose2) {
         Matrix4x4 m1 = Matrix4x4(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

         Matrix4x4 transposed = m1.Transpose();

         Matrix4x4 retransposed = transposed.Transpose();

         EXPECT_EQ(m1, retransposed);
      }
   }
}