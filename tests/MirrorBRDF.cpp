//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "gtest/gtest.h"

#include "../src/structures/Vector.h"
#include "../src/structures/Normal.h"
#include "../src/shading/brdf/MirrorBRDF.h"


namespace Tests {

   using namespace Polytope;

   namespace MirrorBRDF {
      TEST(Transform, GetVectorInPDF1) {

         const Vector incoming(0, 0, -1);
         const Normal normal(0, 0, 1);
         const Vector expected(0, 0, 1);

         Polytope::MirrorBRDF brdf = Polytope::MirrorBRDF();

         float pdf = 0.0;

         Vector actual = brdf.getVectorInPDF(incoming, pdf);

         EXPECT_EQ(expected, actual);
      }

      TEST(Transform, GetVectorInPDF2) {

         Vector incoming(1, -1, 0);
         Normal normal(0, 1, 0);
         Vector expected(1, 1, 0);

         incoming.Normalize();
         normal.Normalize();
         expected.Normalize();

         Polytope::MirrorBRDF brdf = Polytope::MirrorBRDF();

         float pdf = 0.0;

         Vector actual = brdf.getVectorInPDF(incoming, pdf);


         EXPECT_EQ(expected, actual);
      }

      TEST(Transform, GetVectorInPDF3) {

         Vector incoming(2, -1, 0);
         Normal normal(0, 1, 0);
         Vector expected(2, 1, 0);

         incoming.Normalize();
         normal.Normalize();
         expected.Normalize();

         Polytope::MirrorBRDF brdf = Polytope::MirrorBRDF();

         float pdf = 0.0;

         Vector actual = brdf.getVectorInPDF(incoming, pdf);


         EXPECT_EQ(expected, actual);
      }

      TEST(Transform, GetVectorInPDF4) {

         Vector incoming(0, -1, 0);
         Normal normal(0, 1, 0);
         Vector expected(0, 1, 0);

         incoming.Normalize();
         normal.Normalize();
         expected.Normalize();

         Polytope::MirrorBRDF brdf = Polytope::MirrorBRDF();

         float pdf = 0.0;

         Vector actual = brdf.getVectorInPDF(incoming, pdf);


         EXPECT_EQ(expected, actual);
      }
   }

}