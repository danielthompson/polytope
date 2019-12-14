//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../src/parsers/PBRTFileParser.h"
#include "../src/samplers/HaltonSampler.h"
#include "../src/samplers/GridSampler.h"
#include "../src/utilities/Logger.h"
#include "../src/films/PNGFilm.h"
#include "../src/filters/BoxFilter.h"
#include "../src/integrators/PathTraceIntegrator.h"
#include "../src/scenes/NaiveScene.h"
#include "../src/cameras/PerspectiveCamera.h"

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
      }
   }
}
