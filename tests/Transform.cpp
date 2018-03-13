//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "gtest/gtest.h"

#include "../src/structures/Vector.h"
#include "../src/structures/Transform.h"

namespace Tests {

    using Polytope::Transform;

    namespace Equality {
        TEST(Transform, Equals) {
           Transform element1 = Transform();
           Transform element2 = Transform();
           EXPECT_EQ(element1, element2);
        }
    }

   namespace HasScale {
      TEST(Transform, DoesntHaveScale1) {
         Transform element = Transform();
         bool actual = element.HasScale();
         EXPECT_FALSE(actual);
      }

      TEST(Transform, DoesntHaveScale2) {
         Transform element = Transform::Scale(1);
         bool actual = element.HasScale();
         EXPECT_FALSE(actual);
      }

      TEST(Transform, DoesntHaveScale3) {
         Transform element = Transform::Scale(1, 1, 1);
         bool actual = element.HasScale();
         EXPECT_FALSE(actual);
      }

      TEST(Transform, DoesntHaveScale4) {
         Polytope::Vector delta = Polytope::Vector(1, 1, 1);
         Transform element = Transform::Scale(delta);
         bool actual = element.HasScale();
         EXPECT_FALSE(actual);
      }

      TEST(Transform, HasScale1) {
         Transform element = Transform::Scale(2);
         bool actual = element.HasScale();
         EXPECT_TRUE(actual);
      }

      TEST(Transform, HasScale2) {
         Transform element = Transform::Scale(2, 2, 2);
         bool actual = element.HasScale();
         EXPECT_TRUE(actual);
      }

      TEST(Transform, HasScale3) {
         Transform element = Transform::Scale(2, 1, 1);
         bool actual = element.HasScale();
         EXPECT_TRUE(actual);
      }

      TEST(Transform, HasScale4) {
         Transform element = Transform::Scale(1.002, 1, 1);
         bool actual = element.HasScale();
         EXPECT_TRUE(actual);
      }

      TEST(Transform, HasScale5) {
         Polytope::Vector delta = Polytope::Vector(0, 1, 1);
         Transform element = Transform::Scale(delta);
         bool actual = element.HasScale();
         EXPECT_TRUE(actual);
      }
   }

}