//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../../src/common/parsers/PBRTFileParser.h"
#include "../../../src/cpu/samplers/samplers.h"
#include "../../../src/common/utilities/Logger.h"
#include "../../../src/cpu/films/PNGFilm.h"
#include "../../../src/cpu/filters/BoxFilter.h"
#include "../../../src/cpu/integrators/PathTraceIntegrator.h"
#include "../../../src/cpu/scenes/Scene.h"
#include "../../../src/cpu/cameras/PerspectiveCamera.h"
#include "../../../src/cpu/shapes/mesh.h"

// TODO - put all test globals somewhere

namespace Tests {

   using poly::PBRTFileParser;

   namespace Parse {
      TEST(PBRTParse, EmptyWorld) {

         auto fp = PBRTFileParser();
         std::string file = "../scenes/minimum-emptyworld.pbrt";
         std::unique_ptr<poly::AbstractRunner> runner = fp.ParseFile(file);

         // ensure nothing is null
         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);
         ASSERT_NE(nullptr, runner->Film);
         ASSERT_NE(nullptr, runner->Film->Filter);
         ASSERT_NE(nullptr, runner->Integrator);
         ASSERT_NE(nullptr, runner->Integrator->Scene);
         ASSERT_NE(nullptr, runner->Integrator->Scene->Camera);
         EXPECT_EQ(7, runner->Integrator->MaxDepth);
         ASSERT_NE(nullptr, runner->Scene);

         // sampler
         std::unique_ptr<poly::AbstractSampler> sampler = std::move(runner->Sampler);
         poly::HaltonSampler *actualSampler = dynamic_cast<poly::HaltonSampler *>(sampler.get());
         ASSERT_NE(nullptr, actualSampler);

         ASSERT_NE(nullptr, runner->Film);

         // filter
         std::unique_ptr<poly::AbstractFilter> filter = std::move(runner->Film->Filter);
         poly::BoxFilter *actualFilter = dynamic_cast<poly::BoxFilter *>(filter.get());
         ASSERT_NE(nullptr, actualFilter);

         // film
         std::unique_ptr<poly::AbstractFilm> film = std::move(runner->Film);
         poly::PNGFilm *actualFilm = dynamic_cast<poly::PNGFilm *>(film.get());
         ASSERT_NE(nullptr, actualFilm);
         EXPECT_EQ("minimum.png", actualFilm->Filename);
         EXPECT_EQ(640, actualFilm->Bounds.x);
         EXPECT_EQ(640, actualFilm->Bounds.y);

         // integrator
         std::unique_ptr<poly::AbstractIntegrator> integrator = std::move(runner->Integrator);
         poly::PathTraceIntegrator *actualIntegrator = dynamic_cast<poly::PathTraceIntegrator *>(integrator.get());
         ASSERT_NE(nullptr, actualIntegrator);

         // integrator's scene
         poly::Scene* scene = actualIntegrator->Scene;
         poly::Scene* actualScene = dynamic_cast<poly::Scene *>(scene);
         ASSERT_NE(nullptr, actualScene);
         EXPECT_EQ(0, actualScene->Lights.size());
         EXPECT_EQ(0, actualScene->Shapes.size());

         // camera
         std::unique_ptr<poly::AbstractCamera> camera = std::move(scene->Camera);
         poly::PerspectiveCamera* actualCamera = dynamic_cast<poly::PerspectiveCamera *>(camera.get());
         ASSERT_NE(nullptr, actualCamera);
         EXPECT_EQ(640, actualCamera->Settings.Bounds.x);
         EXPECT_EQ(640, actualCamera->Settings.Bounds.y);

         ASSERT_NE(nullptr, actualCamera->CameraToWorld.Matrix.Matrix);

         const auto actualMatrix = actualCamera->CameraToWorld.Matrix.Matrix;

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

         EXPECT_EQ(50, actualCamera->Settings.FieldOfView);
      }

      TEST(PBRTParse, EmptyWorldMissingCamera) {

         PBRTFileParser fp = PBRTFileParser();
         std::string file = "../scenes/minimum-emptyworld-missingcamera.pbrt";
         std::unique_ptr<poly::AbstractRunner> runner = fp.ParseFile(file);

         // ensure nothing is null
         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);
         ASSERT_NE(nullptr, runner->Film);
         ASSERT_NE(nullptr, runner->Film->Filter);
         ASSERT_NE(nullptr, runner->Integrator);
         ASSERT_NE(nullptr, runner->Integrator->Scene);
         ASSERT_NE(nullptr, runner->Integrator->Scene->Camera);
         ASSERT_NE(nullptr, runner->Scene);

         // sampler
         std::unique_ptr<poly::AbstractSampler> sampler = std::move(runner->Sampler);
         poly::HaltonSampler *actualSampler = dynamic_cast<poly::HaltonSampler *>(sampler.get());
         ASSERT_NE(nullptr, actualSampler);

         // filter
         std::unique_ptr<poly::AbstractFilter> filter = std::move(runner->Film->Filter);
         poly::BoxFilter *actualFilter = dynamic_cast<poly::BoxFilter *>(filter.get());
         ASSERT_NE(nullptr, actualFilter);

         // film
         std::unique_ptr<poly::AbstractFilm> film = std::move(runner->Film);
         poly::PNGFilm *actualFilm = dynamic_cast<poly::PNGFilm *>(film.get());
         ASSERT_NE(nullptr, actualFilm);
         EXPECT_EQ("minimum.png", actualFilm->Filename);
         EXPECT_EQ(640, actualFilm->Bounds.x);
         EXPECT_EQ(640, actualFilm->Bounds.y);

         // integrator
         std::unique_ptr<poly::AbstractIntegrator> integrator = std::move(runner->Integrator);
         poly::PathTraceIntegrator *actualIntegrator = dynamic_cast<poly::PathTraceIntegrator *>(integrator.get());
         ASSERT_NE(nullptr, actualIntegrator);
         EXPECT_EQ(7, actualIntegrator->MaxDepth);


         // integrator's scene
         poly::Scene* scene = actualIntegrator->Scene;
         poly::Scene* actualScene = dynamic_cast<poly::Scene *>(scene);
         ASSERT_NE(nullptr, actualScene);
         EXPECT_EQ(0, actualScene->Lights.size());
         EXPECT_EQ(0, actualScene->Shapes.size());

         // camera
         std::unique_ptr<poly::AbstractCamera> camera = std::move(scene->Camera);
         poly::PerspectiveCamera* actualCamera = dynamic_cast<poly::PerspectiveCamera *>(camera.get());
         ASSERT_NE(nullptr, actualCamera);
         EXPECT_EQ(640, actualCamera->Settings.Bounds.x);
         EXPECT_EQ(640, actualCamera->Settings.Bounds.y);

         ASSERT_NE(nullptr, actualCamera->CameraToWorld.Matrix.Matrix);

         const auto actualMatrix = actualCamera->CameraToWorld.Matrix.Matrix;

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

         EXPECT_EQ(90, actualCamera->Settings.FieldOfView);
      }
   }
}
