//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../src/parsers/PBRTFileParser.h"
#include "../src/samplers/HaltonSampler.h"
#include "../src/utilities/Logger.h"
#include "../src/films/PNGFilm.h"
#include "../src/filters/BoxFilter.h"
#include "../src/integrators/PathTraceIntegrator.h"
#include "../src/scenes/NaiveScene.h"
#include "../src/cameras/PerspectiveCamera.h"

// TODO - put all test globals somewhere
Polytope::Logger Log;

namespace Tests {

   using Polytope::PBRTFileParser;

   namespace Parse {
      TEST(FileParser, EmptyWorld) {

         PBRTFileParser fp = PBRTFileParser();
         std::string file = "../scenes/minimum-emptyworld.pbrt";
         std::unique_ptr<Polytope::AbstractRunner> runner = fp.ParseFile(file);

         // ensure nothing is null
         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);
         ASSERT_NE(nullptr, runner->Film);
         ASSERT_NE(nullptr, runner->Film->Filter);
         ASSERT_NE(nullptr, runner->Integrator);
         ASSERT_NE(nullptr, runner->Integrator->Scene);
         ASSERT_NE(nullptr, runner->Integrator->Scene->Camera);
         ASSERT_EQ(7, runner->Integrator->MaxDepth);
         ASSERT_NE(nullptr, runner->Scene);

         // sampler
         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);
         Polytope::HaltonSampler *actualSampler = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());
         ASSERT_NE(nullptr, actualSampler);

         ASSERT_NE(nullptr, runner->Film);

         // filter
         std::unique_ptr<Polytope::AbstractFilter> filter = std::move(runner->Film->Filter);
         Polytope::BoxFilter *actualFilter = dynamic_cast<Polytope::BoxFilter *>(filter.get());
         ASSERT_NE(nullptr, actualFilter);

         // film
         std::unique_ptr<Polytope::AbstractFilm> film = std::move(runner->Film);
         Polytope::PNGFilm *actualFilm = dynamic_cast<Polytope::PNGFilm *>(film.get());
         ASSERT_NE(nullptr, actualFilm);
         ASSERT_EQ("minimum.png", actualFilm->Filename);
         ASSERT_EQ(640, actualFilm->Bounds.x);
         ASSERT_EQ(640, actualFilm->Bounds.y);

         // integrator
         std::unique_ptr<Polytope::AbstractIntegrator> integrator = std::move(runner->Integrator);
         Polytope::PathTraceIntegrator *actualIntegrator = dynamic_cast<Polytope::PathTraceIntegrator *>(integrator.get());
         ASSERT_NE(nullptr, actualIntegrator);

         // integrator's scene
         Polytope::AbstractScene* scene = actualIntegrator->Scene;
         Polytope::NaiveScene* actualScene = dynamic_cast<Polytope::NaiveScene *>(scene);
         ASSERT_NE(nullptr, actualScene);
         ASSERT_EQ(0, actualScene->Lights.size());
         ASSERT_EQ(0, actualScene->Shapes.size());

         // camera
         std::unique_ptr<Polytope::AbstractCamera> camera = std::move(scene->Camera);
         Polytope::PerspectiveCamera* actualCamera = dynamic_cast<Polytope::PerspectiveCamera *>(camera.get());
         ASSERT_NE(nullptr, actualCamera);
         ASSERT_EQ(640, actualCamera->Settings.Bounds.x);
         ASSERT_EQ(640, actualCamera->Settings.Bounds.y);

         ASSERT_NE(nullptr, actualCamera->CameraToWorld.Matrix.Matrix);

         const auto actualMatrix = actualCamera->CameraToWorld.Matrix.Matrix;

         ASSERT_EQ(1, actualMatrix[0][0]);
         ASSERT_EQ(0, actualMatrix[0][1]);
         ASSERT_EQ(0, actualMatrix[0][2]);
         ASSERT_EQ(0, actualMatrix[0][3]);

         ASSERT_EQ(0, actualMatrix[1][0]);
         ASSERT_EQ(1, actualMatrix[1][1]);
         ASSERT_EQ(0, actualMatrix[1][2]);
         ASSERT_EQ(0, actualMatrix[1][3]);

         ASSERT_EQ(0, actualMatrix[2][0]);
         ASSERT_EQ(0, actualMatrix[2][1]);
         ASSERT_EQ(1, actualMatrix[2][2]);
         ASSERT_EQ(0, actualMatrix[2][3]);

         ASSERT_EQ(0, actualMatrix[3][0]);
         ASSERT_EQ(0, actualMatrix[3][1]);
         ASSERT_EQ(0, actualMatrix[3][2]);
         ASSERT_EQ(1, actualMatrix[3][3]);

         ASSERT_EQ(50, actualCamera->Settings.FieldOfView);
      }

      TEST(FileParser, EmptyWorldMissingCamera) {

         PBRTFileParser fp = PBRTFileParser();
         std::string file = "../scenes/minimum-emptyworld-missingcamera.pbrt";
         std::unique_ptr<Polytope::AbstractRunner> runner = fp.ParseFile(file);

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
         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);
         Polytope::HaltonSampler *actualSampler = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());
         ASSERT_NE(nullptr, actualSampler);

         // filter
         std::unique_ptr<Polytope::AbstractFilter> filter = std::move(runner->Film->Filter);
         Polytope::BoxFilter *actualFilter = dynamic_cast<Polytope::BoxFilter *>(filter.get());
         ASSERT_NE(nullptr, actualFilter);

         // film
         std::unique_ptr<Polytope::AbstractFilm> film = std::move(runner->Film);
         Polytope::PNGFilm *actualFilm = dynamic_cast<Polytope::PNGFilm *>(film.get());
         ASSERT_NE(nullptr, actualFilm);
         ASSERT_EQ("minimum.png", actualFilm->Filename);
         ASSERT_EQ(640, actualFilm->Bounds.x);
         ASSERT_EQ(640, actualFilm->Bounds.y);

         // integrator
         std::unique_ptr<Polytope::AbstractIntegrator> integrator = std::move(runner->Integrator);
         Polytope::PathTraceIntegrator *actualIntegrator = dynamic_cast<Polytope::PathTraceIntegrator *>(integrator.get());
         ASSERT_NE(nullptr, actualIntegrator);
         ASSERT_EQ(7, actualIntegrator->MaxDepth);


         // integrator's scene
         Polytope::AbstractScene* scene = actualIntegrator->Scene;
         Polytope::NaiveScene* actualScene = dynamic_cast<Polytope::NaiveScene *>(scene);
         ASSERT_NE(nullptr, actualScene);
         ASSERT_EQ(0, actualScene->Lights.size());
         ASSERT_EQ(0, actualScene->Shapes.size());

         // camera
         std::unique_ptr<Polytope::AbstractCamera> camera = std::move(scene->Camera);
         Polytope::PerspectiveCamera* actualCamera = dynamic_cast<Polytope::PerspectiveCamera *>(camera.get());
         ASSERT_NE(nullptr, actualCamera);
         ASSERT_EQ(640, actualCamera->Settings.Bounds.x);
         ASSERT_EQ(640, actualCamera->Settings.Bounds.y);

         ASSERT_NE(nullptr, actualCamera->CameraToWorld.Matrix.Matrix);

         const auto actualMatrix = actualCamera->CameraToWorld.Matrix.Matrix;

         ASSERT_EQ(1, actualMatrix[0][0]);
         ASSERT_EQ(0, actualMatrix[0][1]);
         ASSERT_EQ(0, actualMatrix[0][2]);
         ASSERT_EQ(0, actualMatrix[0][3]);

         ASSERT_EQ(0, actualMatrix[1][0]);
         ASSERT_EQ(1, actualMatrix[1][1]);
         ASSERT_EQ(0, actualMatrix[1][2]);
         ASSERT_EQ(0, actualMatrix[1][3]);

         ASSERT_EQ(0, actualMatrix[2][0]);
         ASSERT_EQ(0, actualMatrix[2][1]);
         ASSERT_EQ(1, actualMatrix[2][2]);
         ASSERT_EQ(0, actualMatrix[2][3]);

         ASSERT_EQ(0, actualMatrix[3][0]);
         ASSERT_EQ(0, actualMatrix[3][1]);
         ASSERT_EQ(0, actualMatrix[3][2]);
         ASSERT_EQ(1, actualMatrix[3][3]);

         ASSERT_EQ(90, actualCamera->Settings.FieldOfView);
      }
   }
}
