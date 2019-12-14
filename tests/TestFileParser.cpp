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

Polytope::Logger Log;



namespace Tests {

   using Polytope::PBRTFileParser;

   namespace Parse {

      const std::string twoballs = "Integrator \"path\" \"integer maxdepth\" [ 7 ] \n"
                                   "LookAt 0 0 0 0 0 -1 0 1 0 \n"
                                   "_sampler \"random\" \"integer pixelsamples\" [ 64 ] \n"
                                   "PixelFilter \"box\" \"float xwidth\" [ 1.000000 ] \"float ywidth\" [ 1.000000 ] \n"
                                   "Film \"image\" \"integer xresolution\" [ 640 ] \"integer yresolution\" [ 640 ] \"string input_filename\" [ \"two-balls.png\" ] \n"
                                   "Camera \"perspective\" \"float fov\" [ 50 ] \n"
                                   "WorldBegin\n"
                                   "\tMakeNamedMaterial \"lambert\" \"string type\" [ \"matte\" ]  \"rgb Kd\" [ 0.164705 0.631372 0.596078 ] \n"
                                   "\tAttributeBegin\n"
                                   "\t\tAreaLightSource \"diffuse\" \"rgb L\" [ 10 10 10 ] \n"
                                   "\t\tTransformBegin\n"
                                   "\t\t\tTranslate 0 0 -300\n"
                                   "\t\t\tShape \"sphere\" \"float radius\" [ 25 ] \n"
                                   "\t\tTransformEnd\n"
                                   "\tAttributeEnd\n"
                                   "\tNamedMaterial \"lambert\" \n"
                                   "\tTransformBegin\n"
                                   "\t\tTranslate 100 0 -200\n"
                                   "\t\tShape \"sphere\" \"float radius\" [ 50 ] \n"
                                   "\tTransformEnd\n"
                                   "\tTransformBegin\n"
                                   "\t\tTranslate -100 0 -200\n"
                                   "\t\tShape \"sphere\" \"float radius\" [ 50 ] \n"
                                   "\tTransformEnd\n"
                                   "WorldEnd";


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
         ASSERT_EQ(7, runner->Integrator->MaxDepth);
         ASSERT_NE(nullptr, runner->Scene);

         // ensure the sampler is halton
         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);
         Polytope::HaltonSampler *actualSampler = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());
         ASSERT_NE(nullptr, actualSampler);

         ASSERT_NE(nullptr, runner->Film);

         // ensure the film's filter is a box filter
         std::unique_ptr<Polytope::AbstractFilter> filter = std::move(runner->Film->Filter);
         Polytope::BoxFilter *actualFilter = dynamic_cast<Polytope::BoxFilter *>(filter.get());
         ASSERT_NE(nullptr, actualFilter);

         // ensure the film is PNG film
         std::unique_ptr<Polytope::AbstractFilm> film = std::move(runner->Film);
         Polytope::PNGFilm *actualFilm = dynamic_cast<Polytope::PNGFilm *>(film.get());
         ASSERT_NE(nullptr, actualFilm);
         ASSERT_EQ("minimum.png", actualFilm->Filename);


         // ensure the scene is a naive scene
         // TODO std::unique_ptr<Polytope::AbstractScene> scene = std::move(runner->Integrator->Scene);
      }
   }
}
