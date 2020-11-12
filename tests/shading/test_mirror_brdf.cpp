//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "gtest/gtest.h"

#include "../../src/cpu/structures/Vectors.h"
#include "../../src/cpu/structures/Vectors.h"
#include "../../src/cpu/shading/brdf/mirror_brdf.h"


namespace Tests {

   using namespace poly;

   namespace MirrorBRDF {
      // TODO fixme
//      TEST(Transform, GetVectorInPDF1) {
//
//         const Vector incoming(0, 0, -1);
//         const Normal normal(0, 0, 1);
//         const Vector expected(0, 0, 1);
//
//         poly::MirrorBRDF brdf = poly::MirrorBRDF();
//
//         float pdf = 0.0;
//
//         Vector actual = brdf.sample(incoming, pdf);
//
//         EXPECT_FLOAT_EQ(expected.x, actual.x);
//         EXPECT_FLOAT_EQ(expected.y, actual.y);
//         EXPECT_FLOAT_EQ(expected.z, actual.z);
//      }

      TEST(Transform, GetVectorInPDF2) {

         Vector incoming(1, -1, 0);
         Normal normal(0, 1, 0);
         Vector expected(1, 1, 0);

         incoming.Normalize();
         normal.Normalize();
         expected.Normalize();

         ReflectanceSpectrum refl;
         poly::MirrorBRDF brdf = poly::MirrorBRDF(refl);
         
         float pdf = 0.0;

         Vector actual = brdf.sample(incoming, 0, 0, refl, pdf);

         EXPECT_EQ(expected, actual);
      }

      TEST(Transform, GetVectorInPDF3) {

         Vector incoming(2, -1, 0);
         Normal normal(0, 1, 0);
         Vector expected(2, 1, 0);

         incoming.Normalize();
         normal.Normalize();
         expected.Normalize();

         ReflectanceSpectrum refl;
         poly::MirrorBRDF brdf = poly::MirrorBRDF(refl);
         
         
         float pdf = 0.0;

         Vector actual = brdf.sample(incoming, 0, 0, refl, pdf);


         EXPECT_EQ(expected, actual);
      }

      TEST(Transform, GetVectorInPDF4) {

         Vector incoming(0, -1, 0);
         Normal normal(0, 1, 0);
         Vector expected(0, 1, 0);

         incoming.Normalize();
         normal.Normalize();
         expected.Normalize();
         
         ReflectanceSpectrum refl;
         poly::MirrorBRDF brdf = poly::MirrorBRDF(refl);
         
         float pdf = 0.0;

         Vector actual = brdf.sample(incoming, 0, 0, refl, pdf);


         EXPECT_EQ(expected, actual);
      }
   }

}