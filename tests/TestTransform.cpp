//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/Transform.h"
#include "../src/cpu/cameras/PerspectiveCamera.h"

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

   TEST(Transform, LookAt1) {

      const Polytope::Point eye = {0, 0, 35};
      const Polytope::Point look_at = {0, 0, -1};
      Polytope::Vector up = {0, 1, 0};

      Transform t = Transform::LookAt(eye, look_at, up);
      
      //Polytope::PerspectiveCamera camera = Polytope::PerspectiveCamera() 
      Polytope::Ray cameraSpaceRay = {{0, 0, 0}, {0, 0, -1}};
      
      Polytope::Ray worldSpaceRay = t.Apply(cameraSpaceRay);
      
      EXPECT_EQ(worldSpaceRay.Origin.x, eye.x);
      EXPECT_EQ(worldSpaceRay.Origin.y, eye.y);
      EXPECT_EQ(worldSpaceRay.Origin.z, eye.z);

      EXPECT_EQ(worldSpaceRay.Direction.x, 0);
      EXPECT_EQ(worldSpaceRay.Direction.y, 0);
      EXPECT_EQ(worldSpaceRay.Direction.z, -1);
   }

   TEST(Transform, LookAt2) {

      const Polytope::Point eye = {0, 0, -35};
      const Polytope::Point look_at = {0, 0, -1};
      Polytope::Vector up = {0, 1, 0};

      Transform t = Transform::LookAt(eye, look_at, up);

      //Polytope::PerspectiveCamera camera = Polytope::PerspectiveCamera() 
      Polytope::Ray cameraSpaceRay = {{0, 0, 0}, {0, 0, -1}};

      Polytope::Ray worldSpaceRay = t.Apply(cameraSpaceRay);

      EXPECT_EQ(worldSpaceRay.Origin.x, eye.x);
      EXPECT_EQ(worldSpaceRay.Origin.y, eye.y);
      EXPECT_EQ(worldSpaceRay.Origin.z, eye.z);

      EXPECT_EQ(worldSpaceRay.Direction.x, 0);
      EXPECT_EQ(worldSpaceRay.Direction.y, 0);
      EXPECT_EQ(worldSpaceRay.Direction.z, 1);
   }
}