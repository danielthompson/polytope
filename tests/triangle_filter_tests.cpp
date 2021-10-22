//
// Created by daniel on 10/16/20.
//

#include <gtest/gtest.h>
#include "../../src/cpu/filters/triangle_filter.h"

namespace Tests {

   TEST(triangle_filter, test1) {

      const poly::point2f location = { 1.5f, 1.5f };
      const poly::Sample sample = { { 1.f, 1.f, 1.f } };
      const poly::bounds bounds = { 2, 2 };
      poly::triangle_filter filter(bounds, 1);
      filter.add_sample(location, sample);
      filter.pre_output();
      
      auto output = filter.output({0, 0});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);

      output = filter.output({0, 1});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);

      output = filter.output({1, 0});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);

      output = filter.output({1, 1});
      EXPECT_EQ(output.r, 1);
      EXPECT_EQ(output.g, 1);
      EXPECT_EQ(output.b, 1);
   }

   TEST(triangle_filter, test2) {

      const poly::point2f location = { 1.f, 0.5f };
      const poly::Sample sample = { { 1.f, 1.f, 1.f } };
      const poly::bounds bounds = { 2, 2 };
      poly::triangle_filter filter(bounds, 1);
      filter.add_sample(location, sample);
      filter.pre_output();

      auto output = filter.output({0, 0});
      EXPECT_EQ(output.r, .5);
      EXPECT_EQ(output.g, .5);
      EXPECT_EQ(output.b, .5);

      output = filter.output({0, 1});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);

      output = filter.output({1, 0});
      EXPECT_EQ(output.r, .5);
      EXPECT_EQ(output.g, .5);
      EXPECT_EQ(output.b, .5);

      output = filter.output({1, 1});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);
   }

   TEST(triangle_filter, test3) {
      const poly::point2f location = { 1.25f, 0.5f };
      const poly::Sample sample = { { 1.f, 1.f, 1.f } };
      const poly::bounds bounds = { 2, 2 };
      poly::triangle_filter filter(bounds, 1);
      filter.add_sample(location, sample);
      filter.pre_output();

      auto output = filter.output({0, 0});
      EXPECT_EQ(output.r, .25);
      EXPECT_EQ(output.g, .25);
      EXPECT_EQ(output.b, .25);

      output = filter.output({0, 1});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);

      output = filter.output({1, 0});
      EXPECT_EQ(output.r, .75);
      EXPECT_EQ(output.g, .75);
      EXPECT_EQ(output.b, .75);

      output = filter.output({1, 1});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);
   }

   TEST(triangle_filter, test4) {
      const poly::point2f location = { 2.75f, 1.11111116f };
      const poly::Sample sample = { { 1.f, 1.f, 1.f } };
      const poly::bounds bounds = { 5, 5 };
      poly::triangle_filter filter(bounds, 1);
      filter.add_sample(location, sample);
      filter.pre_output();

      auto output = filter.output({1, 0});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);

      output = filter.output({1, 1});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);

      output = filter.output({1, 2});
      EXPECT_EQ(output.r, 0);
      EXPECT_EQ(output.g, 0);
      EXPECT_EQ(output.b, 0);
   }
}