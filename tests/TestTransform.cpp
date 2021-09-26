//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "gtest/gtest.h"

#include "../src/cpu/structures/Vectors.h"
#include "../src/cpu/structures/transform.h"
#include "../src/cpu/cameras/perspective_camera.h"

namespace Tests {

    using poly::transform;

    namespace Equality {
        TEST(Transform, Equals) {
           transform element1 = transform();
           transform element2 = transform();
           EXPECT_EQ(element1, element2);
        }
    }

   namespace HasScale {
      TEST(Transform, DoesntHaveScale1) {
         transform element = transform();
         bool actual = element.has_scale();
         EXPECT_FALSE(actual);
      }

      TEST(Transform, DoesntHaveScale2) {
         transform element = transform::scale(1);
         bool actual = element.has_scale();
         EXPECT_FALSE(actual);
      }

      TEST(Transform, DoesntHaveScale3) {
         transform element = transform::scale(1, 1, 1);
         bool actual = element.has_scale();
         EXPECT_FALSE(actual);
      }

      TEST(Transform, DoesntHaveScale4) {
         poly::vector delta = poly::vector(1, 1, 1);
         transform element = transform::scale(delta);
         bool actual = element.has_scale();
         EXPECT_FALSE(actual);
      }

      TEST(Transform, HasScale1) {
         transform element = transform::scale(2);
         bool actual = element.has_scale();
         EXPECT_TRUE(actual);
      }

      TEST(Transform, HasScale2) {
         transform element = transform::scale(2, 2, 2);
         bool actual = element.has_scale();
         EXPECT_TRUE(actual);
      }

      TEST(Transform, HasScale3) {
         transform element = transform::scale(2, 1, 1);
         bool actual = element.has_scale();
         EXPECT_TRUE(actual);
      }

      TEST(Transform, HasScale4) {
         transform element = transform::scale(1.002, 1, 1);
         bool actual = element.has_scale();
         EXPECT_TRUE(actual);
      }

      TEST(Transform, HasScale5) {
         poly::vector delta = poly::vector(0, 1, 1);
         transform element = transform::scale(delta);
         bool actual = element.has_scale();
         EXPECT_TRUE(actual);
      }
   }

   TEST(Transform, LookAt1) {

      const poly::point eye = {0, 0, 35};
      const poly::point look_at = {0, 0, -1};
      poly::vector up = {0, 1, 0};

      transform t = transform::look_at(eye, look_at, up);
      
      //poly::PerspectiveCamera camera = poly::PerspectiveCamera() 
      poly::ray cameraSpaceRay = {{0, 0, 0}, {0, 0, -1}};
      
      poly::ray worldSpaceRay = t.apply(cameraSpaceRay);
      
      EXPECT_EQ(worldSpaceRay.origin.x, eye.x);
      EXPECT_EQ(worldSpaceRay.origin.y, eye.y);
      EXPECT_EQ(worldSpaceRay.origin.z, eye.z);

      EXPECT_EQ(worldSpaceRay.direction.x, 0);
      EXPECT_EQ(worldSpaceRay.direction.y, 0);
      EXPECT_EQ(worldSpaceRay.direction.z, -1);
   }

   TEST(Transform, LookAt2) {

      const poly::point eye = {0, 0, -35};
      const poly::point look_at = {0, 0, -1};
      poly::vector up = {0, 1, 0};

      transform t = transform::look_at(eye, look_at, up);

      //poly::PerspectiveCamera camera = poly::PerspectiveCamera() 
      poly::ray cameraSpaceRay = {{0, 0, 0}, {0, 0, -1}};

      poly::ray worldSpaceRay = t.apply(cameraSpaceRay);

      EXPECT_EQ(worldSpaceRay.origin.x, eye.x);
      EXPECT_EQ(worldSpaceRay.origin.y, eye.y);
      EXPECT_EQ(worldSpaceRay.origin.z, eye.z);

      EXPECT_EQ(worldSpaceRay.direction.x, 0);
      EXPECT_EQ(worldSpaceRay.direction.y, 0);
      EXPECT_EQ(worldSpaceRay.direction.z, 1);
   }
}