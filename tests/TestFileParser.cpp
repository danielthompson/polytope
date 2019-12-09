//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../src/parsers/PBRTFileParser.h"
#include "../src/samplers/HaltonSampler.h"
#include "../src/samplers/GridSampler.h"

namespace Tests {

   using Polytope::PBRTFileParser;

   namespace Parse {

      TEST(FileParser, Sampler) {

         Polytope::Logger logger = Polytope::Logger();

         PBRTFileParser fp = PBRTFileParser(logger);
         std::string desc = R"(Sampler "random" "integer pixelsamples" [ 64 ] )";
         std::unique_ptr<Polytope::AbstractRunner> runner = fp.ParseString(desc);

         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);

         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);

         Polytope::HaltonSampler *result = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());

         ASSERT_NE(nullptr, result);
      }

      TEST(FileParser, ExampleFile) {

         Polytope::Logger logger = Polytope::Logger();

         PBRTFileParser fp = PBRTFileParser(logger);
         std::string file = "../scenes/./example.pbrt";
         std::unique_ptr<Polytope::AbstractRunner> runner = fp.ParseFile(file);

         ASSERT_NE(nullptr, runner);
         ASSERT_NE(nullptr, runner->Sampler);

         std::unique_ptr<Polytope::AbstractSampler> sampler = std::move(runner->Sampler);

         Polytope::HaltonSampler *result = dynamic_cast<Polytope::HaltonSampler *>(sampler.get());

         ASSERT_NE(nullptr, result);
      }
   }
}
