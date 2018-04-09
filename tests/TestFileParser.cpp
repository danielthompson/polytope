//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../src/parsers/PBRTFileParser.h"

namespace Tests {

   using Polytope::PBRTFileParser;

   namespace Parse {
      TEST(FileParser, Open) {

         const std::string filename = "two-balls.pbrt";

         PBRTFileParser fp = PBRTFileParser(filename);
         fp.Parse();
      }

      TEST(FileParser, Open2) {

         const std::string filename = "two-balls.pbrt";

         PBRTFileParser fp = PBRTFileParser(filename);
         fp.Parse();
      }
   }
}
