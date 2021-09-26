#include "gtest/gtest.h"

#include "../../src/cpu/structures/Vectors.h"
#include "../../src/cpu/constants.h"
#include "../../src/cpu/structures/intersection.h"
#include "../../src/cpu/shapes/mesh.h"

namespace Tests {
   
   std::shared_ptr<poly::mesh_geometry> one_triangle_xy(int fv0, int fv1, int fv2) {
      std::shared_ptr<poly::mesh_geometry> geometry = std::make_shared<poly::mesh_geometry>();
      geometry->x_packed.push_back(0);
      geometry->y_packed.push_back(0);
      geometry->z_packed.push_back(0);

      geometry->x_packed.push_back(0);
      geometry->y_packed.push_back(1);
      geometry->z_packed.push_back(0);

      geometry->x_packed.push_back(1);
      geometry->y_packed.push_back(0);
      geometry->z_packed.push_back(0);

      geometry->fv0.push_back(fv0);
      geometry->fv1.push_back(fv1);
      geometry->fv2.push_back(fv2);

      geometry->num_vertices_packed = 3;
      geometry->num_faces = 1;
      geometry->unpack_faces();
      
      return geometry;
   }
   
   void test_helper(float fv0, float fv1, float fv2, float o_z, float d_z, float n_z) {
      std::shared_ptr<poly::transform> identity = std::make_shared<poly::transform>();
      std::shared_ptr<poly::mesh_geometry> geometry = one_triangle_xy(fv0, fv1, fv2); 
      
      // hits, from front

      poly::ray ray(poly::point(0.2f, 0.2f, o_z), poly::vector(0, 0, d_z));
      poly::intersection intersection;

      unsigned int face_index = 0;
      poly::Mesh mesh(identity, identity, nullptr, geometry);

      mesh.intersect(ray, intersection, &face_index, geometry->num_faces);
      EXPECT_TRUE(intersection.Hits);
      EXPECT_EQ(&mesh, intersection.shape);
      
      EXPECT_FLOAT_EQ(0, intersection.geo_normal.x);
      EXPECT_FLOAT_EQ(0, intersection.geo_normal.y);
      EXPECT_FLOAT_EQ(n_z, intersection.geo_normal.z);
      
      EXPECT_FLOAT_EQ(.2f, intersection.location.x);
      EXPECT_FLOAT_EQ(.2f, intersection.location.y);
      EXPECT_FLOAT_EQ(0, intersection.location.z);
   }


   TEST(mesh_cpu, intersect_one_front_cw) {
      test_helper(0, 1, 2, -10, 1, -1);
   }
   
   TEST(mesh_cpu, intersect_one_back_cw) {
      test_helper(0, 2, 1, -10, 1, -1);
   }

   TEST(mesh_cpu, intersect_one_front_ccw) {
      test_helper(0, 1, 2, 10, -1, 1);
   }

   TEST(mesh_cpu, intersect_one_back_ccw) {
      test_helper(0, 2, 1, 10, -1, 1);
   }
   
   TEST(mesh_cpu, intersect_stay_above_surface) {
      std::shared_ptr<poly::transform> identity = std::make_shared<poly::transform>();
      std::shared_ptr<poly::mesh_geometry> geometry = one_triangle_xy(0, 1, 2);
      float z = .2f;
      geometry->z[0] = z;
      geometry->z[1] = z;
      geometry->z[2] = z;
      geometry->z_packed[0] = z;
      geometry->z_packed[1] = z;
      geometry->z_packed[2] = z;

      poly::point expected_hit_point = {0.2f, .2f, z};
      
      for (int k = 100; k < 10000000; k++) {
//         for (int j = -50; j < 50; j++) {
//            for (int i = -50; i < 50; i++) {
               poly::point origin = {1.464552f, 3.635245f, -1.0364f * (float)k};
               poly::vector direction = expected_hit_point - origin;
         direction.normalize();
               poly::ray ray(origin, direction);
               poly::intersection intersection;

               unsigned int face_index = 0;
               poly::Mesh mesh(identity, identity, nullptr, geometry);

               mesh.intersect(ray, intersection, &face_index, geometry->num_faces);
               ASSERT_TRUE(intersection.Hits);
               ASSERT_EQ(&mesh, intersection.shape);

               ASSERT_FLOAT_EQ(0, intersection.geo_normal.x);
               ASSERT_FLOAT_EQ(0, intersection.geo_normal.y);
               ASSERT_FLOAT_EQ(-1, intersection.geo_normal.z);

//               EXPECT_FLOAT_EQ(expected_hit_point.x, intersection.Location.x);
//               EXPECT_FLOAT_EQ(expected_hit_point.y, intersection.Location.y);
               if (k == 9999999)
                  ASSERT_FLOAT_EQ(expected_hit_point.z, intersection.location.z);
            }
//         }
//      }
   }
}
