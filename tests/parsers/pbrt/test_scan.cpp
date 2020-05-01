//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../../src/parsers/PBRTFileParser.h"
#include "../../../src/samplers/samplers.h"
#include "../../../src/utilities/Logger.h"
#include "../../../src/films/PNGFilm.h"
#include "../../../src/filters/BoxFilter.h"
#include "../../../src/integrators/PathTraceIntegrator.h"
#include "../../../src/scenes/NaiveScene.h"
#include "../../../src/cameras/PerspectiveCamera.h"

// TODO - put all test globals somewhere
Polytope::Logger Log;

namespace Tests {

   using Polytope::PBRTFileParser;
   
   namespace Scan {
      
      std::vector<std::string> test_scan(const std::string &line, const int expected_tokens) {
         PBRTFileParser parser = PBRTFileParser();

         const std::unique_ptr<std::istream> stream = std::make_unique<std::istringstream>(line);
         const auto tokenized_lines = PBRTFileParser::Scan(stream);

         EXPECT_EQ(tokenized_lines->size(), 1);

         std::vector<std::string> tokens = (*tokenized_lines)[0];

         EXPECT_EQ(tokens.size(), expected_tokens);
         
         return tokens;
      }
      
      void expect_single_value(const std::vector<std::string> &tokens) {
         EXPECT_EQ(tokens[0], "WorldBegin");
         EXPECT_EQ(tokens[1], R"("name")");
         EXPECT_EQ(tokens[2], R"("param_type)");
         EXPECT_EQ(tokens[3], R"(param_name")");
         EXPECT_EQ(tokens[4], "1");
      }
      
      TEST(PBRTScan, SingleValueInBracketsWithSpace) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" [ 1 ])", 5);
         expect_single_value(tokens);
      }

      TEST(PBRTScan, SingleValueInBracketsWithoutSpace) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" [1])", 5);
         expect_single_value(tokens);
      }

      TEST(PBRTScan, SingleValueNotInBrackets) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" 1)", 5);
         expect_single_value(tokens);
      }

      TEST(PBRTScan, SingleValueInBracketsWithLeftSpace) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" [ 1])", 5);
         expect_single_value(tokens);
      }

      TEST(PBRTScan, SingleValueInBracketsWithRightSpace) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" [1 ])", 5);
         expect_single_value(tokens);
      }

      void expect_array_value(const std::vector<std::string> &tokens) {
         EXPECT_EQ(tokens[0], "WorldBegin");
         EXPECT_EQ(tokens[1], R"("name")");
         EXPECT_EQ(tokens[2], R"("param_type)");
         EXPECT_EQ(tokens[3], R"(param_name")");
         EXPECT_EQ(tokens[4], "[");
         EXPECT_EQ(tokens[5], "12.3");
         EXPECT_EQ(tokens[6], "45.6");
         EXPECT_EQ(tokens[7], "78.9");
         EXPECT_EQ(tokens[8], "]");
      }
      
      TEST(PBRTScan, ArrayValueWithoutSpace) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" [12.3 45.6 78.9])", 9);
         expect_array_value(tokens);
      }

      TEST(PBRTScan, ArrayValueWithSpace) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" [ 12.3 45.6 78.9 ])", 9);
         expect_array_value(tokens);
      }

      TEST(PBRTScan, ArrayValueWithLeftSpace) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" [ 12.3 45.6 78.9])", 9);
         expect_array_value(tokens);
      }

      TEST(PBRTScan, ArrayValueWithRightSpace) {
         const std::vector<std::string> tokens = test_scan(R"(WorldBegin "name" "param_type param_name" [12.3 45.6 78.9 ])", 9);
         expect_array_value(tokens);
      }
   }
}
