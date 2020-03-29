#include "gtest/gtest.h"

#include "../src/structures/Vectors.h"
#include "../src/Constants.h"
#include "../src/shapes/triangle.h"
#include "../src/structures/Intersection.h"

namespace Tests {

   namespace TriangleMeshISPC {

      TEST(TriangleMeshISPC, Hits) {
         std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::TriangleMeshSOA tm(identity, identity, nullptr);

         // tm.Vertices.emplace_back(Polytope::Point(0, 0, 0));
         tm.x.push_back(0);
         tm.y.push_back(0);
         tm.z.push_back(0);

         // tm.Vertices.emplace_back(Polytope::Point(0, 1, 0));
         tm.x.push_back(0);
         tm.y.push_back(1);
         tm.z.push_back(0);

         // tm.Vertices.emplace_back(Polytope::Point(1, 0, 0));
         tm.x.push_back(1);
         tm.y.push_back(0);
         tm.z.push_back(0);

         // tm.Faces.emplace_back(Polytope::Point3ui(0, 1, 2));
         tm.fv0.push_back(0);
         tm.fv1.push_back(1);
         tm.fv2.push_back(2);
         
         tm.num_faces = 1;
         tm.num_vertices = 3;
         
         // hits, from either direction

         {
            Polytope::Ray ray(0.2f, 0.2f, -10, 0, 0, 1);
            bool hits = tm.Hits(ray);
            EXPECT_TRUE(hits);
         }

         {
            Polytope::Ray ray(0.2f, 0.2f, 10, 0, 0, -1);
            bool hits = tm.Hits(ray);
            EXPECT_TRUE(hits);
         }

         {
            Polytope::Ray ray(0.2f, 0.2f, 10, 0, 0, 1);
            bool hits = tm.Hits(ray);
            EXPECT_FALSE(hits);
         }

         {
            Polytope::Ray ray(0.2f, 0.2f, -10, 0, 0, -1);
            bool hits = tm.Hits(ray);
            EXPECT_FALSE(hits);
         }

         {
            Polytope::Ray ray(0.2f, 0.2f, -10, 0, 1, 0);
            bool hits = tm.Hits(ray);
            EXPECT_FALSE(hits);
         }
      }

      TEST(TriangleMeshISPC, Intersects) {
         std::shared_ptr<Polytope::Transform> identity = std::make_shared<Polytope::Transform>();
         Polytope::TriangleMeshSOA tm(identity, identity, nullptr);

         // face 0
         
         // tm.Vertices.emplace_back(Polytope::Point(0, 0, 0));
         tm.x.push_back(0);
         tm.y.push_back(0);
         tm.z.push_back(0);

         // tm.Vertices.emplace_back(Polytope::Point(0, 1, 0));
         tm.x.push_back(0);
         tm.y.push_back(1);
         tm.z.push_back(0);

         // tm.Vertices.emplace_back(Polytope::Point(1, 0, 0));
         tm.x.push_back(1);
         tm.y.push_back(0);
         tm.z.push_back(0);

         // tm.Faces.emplace_back(Polytope::Point3ui(0, 1, 2));
         tm.fv0.push_back(0);
         tm.fv1.push_back(1);
         tm.fv2.push_back(2);

         // face 1

         //tm.Vertices.emplace_back(Polytope::Point(0, 0, 1));
         tm.x.push_back(0);
         tm.y.push_back(0);
         tm.z.push_back(1);

         // tm.Vertices.emplace_back(Polytope::Point(0, 1, 1));
         tm.x.push_back(0);
         tm.y.push_back(1);
         tm.z.push_back(1);

         // tm.Vertices.emplace_back(Polytope::Point(1, 0, 1));
         tm.x.push_back(1);
         tm.y.push_back(0);
         tm.z.push_back(1);

         // tm.Faces.emplace_back(Polytope::Point3ui(3, 4, 5));
         tm.fv0.push_back(3);
         tm.fv1.push_back(4);
         tm.fv2.push_back(5);

         tm.num_vertices = 6;
         tm.num_faces = 2;
         
         // hits, from either direction

         {
            Polytope::Ray ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, 1));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_TRUE(intersection.Hits);
            EXPECT_EQ(&tm, intersection.Shape);
            EXPECT_EQ(Polytope::Normal(0, 0, -1), intersection.Normal);
            EXPECT_EQ(Polytope::Point(0.2f, 0.2f, 0), intersection.Location);
         }

         {
            Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, -1));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_TRUE(intersection.Hits);
            EXPECT_EQ(&tm, intersection.Shape);
            EXPECT_EQ(Polytope::Normal(0, 0, 1), intersection.Normal);
            EXPECT_EQ(Polytope::Point(0.2f, 0.2f, 1), intersection.Location);
         }

         {
            Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 10), Polytope::Vector(0, 0, 1));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }

         {
            Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, -10), Polytope::Vector(0, 0, -1));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }

         // parallel
         {
            Polytope::Ray ray = Polytope::Ray(Polytope::Point(0.2f, 0.2f, 0.5f), Polytope::Vector(0, 1, 0));
            Polytope::Intersection intersection;
            tm.Intersect(ray, &intersection);
            EXPECT_FALSE(intersection.Hits);
         }
      }
   }
}
