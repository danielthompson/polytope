//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../../src/common/parsers/pbrt_parser.h"
#include "../../../src/cpu/samplers/samplers.h"
#include "../../../src/common/utilities/Logger.h"
#include "../../../src/cpu/films/PNGFilm.h"
#include "../../../src/cpu/filters/box_filter.h"
#include "../../../src/cpu/integrators/PathTraceIntegrator.h"
#include "../../../src/cpu/scenes/scene.h"
#include "../../../src/cpu/cameras/perspective_camera.h"
#include "../../../src/cpu/shapes/mesh.h"

// TODO - put all test globals somewhere

namespace Tests {

   using poly::pbrt_parser;

   namespace Parse {
      TEST(PBRTParse, EmptyWorld) {

         auto fp = pbrt_parser();
         std::string file = "../scenes/minimum-emptyworld.pbrt";
         std::shared_ptr<poly::runner> runner = fp.parse_file(file);

         // ensure nothing is null
         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);
         ASSERT_NE(nullptr, runner->Film);
         ASSERT_NE(nullptr, runner->Film->Filter);
         ASSERT_NE(nullptr, runner->integrator);
         ASSERT_NE(nullptr, runner->integrator->Scene);
         ASSERT_NE(nullptr, runner->integrator->Scene->Camera);
         EXPECT_EQ(7, runner->integrator->MaxDepth);
         ASSERT_NE(nullptr, runner->Scene);

         // sampler
         std::unique_ptr<poly::abstract_sampler> sampler = std::move(runner->Sampler);
         poly::HaltonSampler *actualSampler = dynamic_cast<poly::HaltonSampler *>(sampler.get());
         ASSERT_NE(nullptr, actualSampler);

         ASSERT_NE(nullptr, runner->Film);

         // filter
         std::unique_ptr<poly::abstract_filter> filter = std::move(runner->Film->Filter);
         poly::box_filter *actualFilter = dynamic_cast<poly::box_filter *>(filter.get());
         ASSERT_NE(nullptr, actualFilter);

         // film
         std::unique_ptr<poly::abstract_film> film = std::move(runner->Film);
         poly::PNGFilm *actualFilm = dynamic_cast<poly::PNGFilm *>(film.get());
         ASSERT_NE(nullptr, actualFilm);
         EXPECT_EQ("minimum.png", actualFilm->Filename);
         EXPECT_EQ(640, actualFilm->Bounds.x);
         EXPECT_EQ(640, actualFilm->Bounds.y);

         // integrator
         std::shared_ptr<poly::abstract_integrator> integrator = std::move(runner->integrator);
         poly::PathTraceIntegrator *actualIntegrator = dynamic_cast<poly::PathTraceIntegrator *>(integrator.get());
         ASSERT_NE(nullptr, actualIntegrator);

         // integrator's scene
         auto scene = actualIntegrator->Scene;
         ASSERT_NE(nullptr, scene);
         EXPECT_EQ(0, scene->Lights.size());
         EXPECT_EQ(0, scene->Shapes.size());

         // camera
         std::unique_ptr<poly::abstract_camera> camera = std::move(scene->Camera);
         poly::perspective_camera* actualCamera = dynamic_cast<poly::perspective_camera *>(camera.get());
         ASSERT_NE(nullptr, actualCamera);
         EXPECT_EQ(640, actualCamera->settings.bounds.x);
         EXPECT_EQ(640, actualCamera->settings.bounds.y);

         ASSERT_NE(nullptr, actualCamera->camera_to_world.matrix.mat);

         const auto actualMatrix = actualCamera->camera_to_world.matrix.mat;

         EXPECT_EQ(-1, actualMatrix[0][0]);
         EXPECT_EQ(0, actualMatrix[0][1]);
         EXPECT_EQ(0, actualMatrix[0][2]);
         EXPECT_EQ(0, actualMatrix[0][3]);

         EXPECT_EQ(0, actualMatrix[1][0]);
         EXPECT_EQ(1, actualMatrix[1][1]);
         EXPECT_EQ(0, actualMatrix[1][2]);
         EXPECT_EQ(0, actualMatrix[1][3]);

         EXPECT_EQ(0, actualMatrix[2][0]);
         EXPECT_EQ(0, actualMatrix[2][1]);
         EXPECT_EQ(-1, actualMatrix[2][2]);
         EXPECT_EQ(0, actualMatrix[2][3]);

         EXPECT_EQ(0, actualMatrix[3][0]);
         EXPECT_EQ(0, actualMatrix[3][1]);
         EXPECT_EQ(0, actualMatrix[3][2]);
         EXPECT_EQ(1, actualMatrix[3][3]);

         EXPECT_EQ(50, actualCamera->settings.field_of_view);
      }

      TEST(PBRTParse, EmptyWorldMissingCamera) {

         pbrt_parser fp = pbrt_parser();
         std::string file = "../scenes/minimum-emptyworld-missingcamera.pbrt";
         std::shared_ptr<poly::runner> runner = fp.parse_file(file);

         // ensure nothing is null
         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);
         ASSERT_NE(nullptr, runner->Film);
         ASSERT_NE(nullptr, runner->Film->Filter);
         ASSERT_NE(nullptr, runner->integrator);
         ASSERT_NE(nullptr, runner->integrator->Scene);
         ASSERT_NE(nullptr, runner->integrator->Scene->Camera);
         ASSERT_NE(nullptr, runner->Scene);

         // sampler
         std::unique_ptr<poly::abstract_sampler> sampler = std::move(runner->Sampler);
         poly::HaltonSampler *actualSampler = dynamic_cast<poly::HaltonSampler *>(sampler.get());
         ASSERT_NE(nullptr, actualSampler);

         // filter
         std::unique_ptr<poly::abstract_filter> filter = std::move(runner->Film->Filter);
         poly::box_filter *actualFilter = dynamic_cast<poly::box_filter *>(filter.get());
         ASSERT_NE(nullptr, actualFilter);

         // film
         std::unique_ptr<poly::abstract_film> film = std::move(runner->Film);
         poly::PNGFilm *actualFilm = dynamic_cast<poly::PNGFilm *>(film.get());
         ASSERT_NE(nullptr, actualFilm);
         EXPECT_EQ("minimum.png", actualFilm->Filename);
         EXPECT_EQ(640, actualFilm->Bounds.x);
         EXPECT_EQ(640, actualFilm->Bounds.y);

         // integrator
         std::shared_ptr<poly::abstract_integrator> integrator = std::move(runner->integrator);
         poly::PathTraceIntegrator *actualIntegrator = dynamic_cast<poly::PathTraceIntegrator *>(integrator.get());
         ASSERT_NE(nullptr, actualIntegrator);
         EXPECT_EQ(7, actualIntegrator->MaxDepth);


         // integrator's scene
         auto scene = actualIntegrator->Scene;
         ASSERT_NE(nullptr, scene);
         EXPECT_EQ(0, scene->Lights.size());
         EXPECT_EQ(0, scene->Shapes.size());

         // camera
         std::unique_ptr<poly::abstract_camera> camera = std::move(scene->Camera);
         poly::perspective_camera* actualCamera = dynamic_cast<poly::perspective_camera *>(camera.get());
         ASSERT_NE(nullptr, actualCamera);
         EXPECT_EQ(640, actualCamera->settings.bounds.x);
         EXPECT_EQ(640, actualCamera->settings.bounds.y);

         ASSERT_NE(nullptr, actualCamera->camera_to_world.matrix.mat);

         const auto actualMatrix = actualCamera->camera_to_world.matrix.mat;

         EXPECT_EQ(-1, actualMatrix[0][0]);
         EXPECT_EQ(0, actualMatrix[0][1]);
         EXPECT_EQ(0, actualMatrix[0][2]);
         EXPECT_EQ(0, actualMatrix[0][3]);

         EXPECT_EQ(0, actualMatrix[1][0]);
         EXPECT_EQ(1, actualMatrix[1][1]);
         EXPECT_EQ(0, actualMatrix[1][2]);
         EXPECT_EQ(0, actualMatrix[1][3]);

         EXPECT_EQ(0, actualMatrix[2][0]);
         EXPECT_EQ(0, actualMatrix[2][1]);
         EXPECT_EQ(-1, actualMatrix[2][2]);
         EXPECT_EQ(0, actualMatrix[2][3]);

         EXPECT_EQ(0, actualMatrix[3][0]);
         EXPECT_EQ(0, actualMatrix[3][1]);
         EXPECT_EQ(0, actualMatrix[3][2]);
         EXPECT_EQ(1, actualMatrix[3][3]);

         EXPECT_EQ(90, actualCamera->settings.field_of_view);
      }
   }
}
