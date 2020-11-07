#include "gtest/gtest.h"

#include "../../src/cpu/structures/Vectors.h"
#include "../../src/cpu/constants.h"
#include "../../src/cpu/structures/Intersection.h"
#include "../../src/cpu/shapes/mesh.h"

namespace Tests {
   TEST(mesh_cpu, intersect_one_front) {
      std::shared_ptr<poly::Transform> identity = std::make_shared<poly::Transform>();
      std::shared_ptr<poly::mesh_geometry> geometry = std::make_shared<poly::mesh_geometry>(); 
      poly::Mesh mesh(identity, identity, nullptr, geometry);

      geometry->x_packed.push_back(0);
      geometry->y_packed.push_back(0);
      geometry->z_packed.push_back(0);

      geometry->x_packed.push_back(0);
      geometry->y_packed.push_back(1);
      geometry->z_packed.push_back(0);

      geometry->x_packed.push_back(1);
      geometry->y_packed.push_back(0);
      geometry->z_packed.push_back(0);

      geometry->fv0.push_back(0);
      geometry->fv1.push_back(1);
      geometry->fv2.push_back(2);

      geometry->num_vertices_packed = 3;
      geometry->num_faces = 1;
      
      geometry->unpack_faces();
      
      // hits, from front

      poly::Ray ray(poly::Point(0.2f, 0.2f, -10), poly::Vector(0, 0, 1));
      poly::Intersection intersection;

      unsigned int face_index = 0;
      
      mesh.intersect(ray, intersection, &face_index, geometry->num_faces);
      EXPECT_TRUE(intersection.Hits);
      EXPECT_EQ(&mesh, intersection.Shape);
      
      EXPECT_FLOAT_EQ(0, intersection.geo_normal.x);
      EXPECT_FLOAT_EQ(0, intersection.geo_normal.y);
      EXPECT_FLOAT_EQ(-1, intersection.geo_normal.z);
      
      EXPECT_FLOAT_EQ(.2f, intersection.Location.x);
      EXPECT_FLOAT_EQ(.2f, intersection.Location.y);
      EXPECT_FLOAT_EQ(0, intersection.Location.z);
      
   }
}
