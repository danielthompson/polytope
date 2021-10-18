//
// Created by daniel on 10/31/20.
//

#include "gtest/gtest.h"

#include "../../../src/common/parsers/pbrt_parser.h"
#include "../../../src/cpu/films/PNGFilm.h"
#include "../../../src/cpu/filters/box_filter.h"
#include "../../../src/cpu/integrators/PathTraceIntegrator.h"
#include "../../../src/cpu/cameras/perspective_camera.h"
#include "../../../src/cpu/shading/brdf/lambert_brdf.h"
#include "../../../lib/lodepng.h"

namespace Tests {
   TEST(parse_texture, imagemap_prelim) {
   
      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/texture/rainbow-texture.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);
   
      // ensure nothing is null
      ASSERT_NE(nullptr, runner->Scene);
      
      ASSERT_EQ(1, runner->Scene->Shapes.size());
      poly::Mesh* mesh = runner->Scene->Shapes[0];
      ASSERT_NE(nullptr, mesh);
      
      auto geometry = mesh->mesh_geometry;
      ASSERT_NE(nullptr, geometry);
      
      ASSERT_FALSE(geometry->has_vertex_normals);
      EXPECT_EQ(1, geometry->num_faces);
      EXPECT_EQ(3, geometry->num_vertices_packed);
      
      ASSERT_NE(nullptr, mesh->material);
      
      ASSERT_NE(nullptr, mesh->material->BRDF);
      
      std::shared_ptr<poly::LambertBRDF> lambert_brdf = std::dynamic_pointer_cast<poly::LambertBRDF>(mesh->material->BRDF);
      std::shared_ptr<poly::texture> texture = lambert_brdf->texture;
      ASSERT_NE(nullptr, texture);
      
      EXPECT_EQ("rainbow", texture->name);
      EXPECT_EQ(873, texture->height);
      EXPECT_EQ(1197, texture->width);
      
      ASSERT_FALSE(texture->data.empty());
      
      ASSERT_EQ(873 * 1197 * 4, texture->data.size());
   }

   TEST(parse_texture, imagemap1) {

      auto fp = poly::pbrt_parser();
      std::string file = "../scenes/test/texture/rainbow-texture.pbrt";
      std::shared_ptr<poly::runner> runner = fp.parse_file(file);

      // ensure nothing is null
   }
   
   TEST(parse_texture, grey) {
      std::vector<unsigned char> v;
      unsigned int width, height;
      unsigned int error = lodepng::decode(v, width, height, "../scenes/pbrt-book/texture/uneven_bump.png", LodePNGColorType::LCT_GREY);
      std::cout << width << " " << height << std::endl;
   }
}