//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../src/parsers/PBRTFileParser.h"
#include "../src/samplers/HaltonSampler.h"
#include "../src/samplers/GridSampler.h"
#include "../src/utilities/Logger.h"

Polytope::Logger Log;



namespace Tests {

   using Polytope::PBRTFileParser;

   namespace Parse {

      const std::string twoballs = "Integrator \"path\" \"integer maxdepth\" [ 7 ] \n"
                                   "LookAt 0 0 0 0 0 -1 0 1 0 \n"
                                   "Sampler \"random\" \"integer pixelsamples\" [ 64 ] \n"
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

      TEST(FileParser, Sampler1) {

         PBRTFileParser fp = PBRTFileParser();
         std::string desc = R"(Sampler "random" "integer pixelsamples" [ 64 ] )";
         std::unique_ptr<Polytope::AbstractRunner> runner = fp.ParseString(desc);

         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);

         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);

         Polytope::HaltonSampler *result = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());

         ASSERT_NE(nullptr, result);
      }

      TEST(FileParser, ExampleFile) {

         PBRTFileParser fp = PBRTFileParser();
         std::string file = "../scenes/./example.pbrt";
         std::unique_ptr<Polytope::AbstractRunner> runner = fp.ParseFile(file);

         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);

         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);

         Polytope::HaltonSampler *result = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());

         ASSERT_NE(nullptr, result);
      }

      TEST(FileParser, Sampler2) {

         PBRTFileParser fp = PBRTFileParser();
         std::string desc = R"(Sampler "halton" "integer pixelsamples" [ 64 ] )";
         std::unique_ptr<Polytope::AbstractRunner> runner = fp.ParseString(desc);

         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);

         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);

         Polytope::HaltonSampler *result = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());

         ASSERT_NE(nullptr, result);

      }

      TEST(FileParser, Sampler3) {

         PBRTFileParser fp = PBRTFileParser();

         std::unique_ptr<Polytope::AbstractRunner> runner = fp.ParseString(twoballs);

         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);

         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);

         Polytope::HaltonSampler *result = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());

         ASSERT_NE(nullptr, result);

      }
   }
}
