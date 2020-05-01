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

namespace Tests {

   using Polytope::PBRTFileParser;
   
   namespace Lex {
      
      std::unique_ptr<Polytope::PBRTDirective> test_lex(const std::vector<std::string> &line_tokens) {
         EXPECT_GT(line_tokens.size(), 0);
         return PBRTFileParser::Lex(line_tokens);
      }

      TEST(PBRTLex, DirectiveOnly) {
         std::vector<std::string> line_tokens = {"directive_name"};
         const std::unique_ptr<Polytope::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "directive_name");
         EXPECT_EQ(directive->Identifier, "");
         EXPECT_EQ(directive->Arguments.size(), 0);
      }

      TEST(PBRTLex, DirectiveWithIdentifierWithoutArgument) {
         std::vector<std::string> line_tokens = {"directive_name", R"("identifier")"};
         const std::unique_ptr<Polytope::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "directive_name");
         EXPECT_EQ(directive->Identifier, "identifier");
         EXPECT_EQ(directive->Arguments.size(), 0);
      }

      TEST(PBRTLex, ScaleDirective1) {
         std::vector<std::string> line_tokens = {"Scale", "1", "2", "3"};
         const std::unique_ptr<Polytope::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "Scale");
         EXPECT_EQ(directive->Identifier, "");
         EXPECT_EQ(directive->Arguments.size(), 1);
         EXPECT_EQ(directive->Arguments[0].Type, Polytope::PBRTArgument::PBRTArgumentType::pbrt_float);
         EXPECT_EQ(directive->Arguments[0].Name, "");
         ASSERT_EQ(directive->Arguments[0].float_values->size(), 3);
         EXPECT_EQ(directive->Arguments[0].float_values->at(0), 1.f);
         EXPECT_EQ(directive->Arguments[0].float_values->at(1), 2.f);
         EXPECT_EQ(directive->Arguments[0].float_values->at(2), 3.f);
      }

      TEST(PBRTLex, ScaleDirective2) {
         std::vector<std::string> line_tokens = {"Scale", "1", "2"};
         try {
            const std::unique_ptr<Polytope::PBRTDirective> directive = test_lex(line_tokens);
            FAIL() << "Expected std::invalid_argument";
         }
         catch (const std::invalid_argument& ex) {
            SUCCEED();
         }
         catch (...) {
            FAIL() << "Expected std::invalid_argument";
         }

      }

      TEST(PBRTLex, ScaleDirective3) {
         std::vector<std::string> line_tokens = {"Scale", "1", "2", "3", "4"};
         try {
            const std::unique_ptr<Polytope::PBRTDirective> directive = test_lex(line_tokens);
            FAIL() << "Expected std::invalid_argument";
         }
         catch (const std::invalid_argument& ex) {
            SUCCEED();
         }
         catch (...) {
            FAIL() << "Expected std::invalid_argument";
         }
      }
      
//      TEST(PBRTLex, DirectiveWithOneArgumentSingle) {
//         std::vector<std::string> line_tokens = {"directive_name", R"("identifier")", R"("param_type)", R"(param_name")", "1"};
//         const std::unique_ptr<Polytope::PBRTDirective> directive = test_lex(line_tokens);
//         ASSERT_NE(directive, nullptr);
//         EXPECT_EQ(directive->Name, "directive_name");
//         EXPECT_EQ(directive->Identifier, "identifier");
//         ASSERT_EQ(directive->Arguments.size(), 1);
//         EXPECT_EQ(directive->Arguments[0].Type, "param_type");
//         EXPECT_EQ(directive->Arguments[0].Name, "param_name");
//         ASSERT_EQ(directive->Arguments[0].Values.size(), 1);
//         EXPECT_EQ(directive->Arguments[0].Values[0], "1");
//      }
   }
}