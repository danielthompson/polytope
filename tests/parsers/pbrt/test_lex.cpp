//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../../src/common/parsers/PBRTFileParser.h"
#include "../../../src/cpu/shapes/mesh.h"

namespace Tests {

   using poly::PBRTFileParser;
   
   namespace Lex {
      
      std::unique_ptr<poly::PBRTDirective> test_lex(const std::vector<std::string> &line_tokens) {
         EXPECT_GT(line_tokens.size(), 0);
         return PBRTFileParser::Lex(line_tokens);
      }

      TEST(PBRTLex, DirectiveOnly) {
         std::vector<std::string> line_tokens = {"directive_name"};
         const std::unique_ptr<poly::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "directive_name");
         EXPECT_EQ(directive->Identifier, "");
         EXPECT_EQ(directive->Arguments.size(), 0);
      }

      TEST(PBRTLex, DirectiveWithIdentifierWithoutArgument) {
         std::vector<std::string> line_tokens = {"directive_name", R"("identifier")"};
         const std::unique_ptr<poly::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "directive_name");
         EXPECT_EQ(directive->Identifier, "identifier");
         EXPECT_EQ(directive->Arguments.size(), 0);
      }

      TEST(PBRTLex, ScaleDirective1) {
         std::vector<std::string> line_tokens = {"Scale", "1", "2", "3"};
         const std::unique_ptr<poly::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "Scale");
         EXPECT_EQ(directive->Identifier, "");
         EXPECT_EQ(directive->Arguments.size(), 1);
         EXPECT_EQ(directive->Arguments[0].Type, poly::PBRTArgument::PBRTArgumentType::pbrt_float);
         EXPECT_EQ(directive->Arguments[0].Name, "");
         ASSERT_EQ(directive->Arguments[0].float_values->size(), 3);
         EXPECT_EQ(directive->Arguments[0].float_values->at(0), 1.f);
         EXPECT_EQ(directive->Arguments[0].float_values->at(1), 2.f);
         EXPECT_EQ(directive->Arguments[0].float_values->at(2), 3.f);
      }

      TEST(PBRTLex, ScaleDirective2) {
         std::vector<std::string> line_tokens = {"Scale", "1", "2"};
         try {
            const std::unique_ptr<poly::PBRTDirective> directive = test_lex(line_tokens);
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
            const std::unique_ptr<poly::PBRTDirective> directive = test_lex(line_tokens);
            FAIL() << "Expected std::invalid_argument";
         }
         catch (const std::invalid_argument& ex) {
            SUCCEED();
         }
         catch (...) {
            FAIL() << "Expected std::invalid_argument";
         }
      }

      TEST(PBRTLex, AreaLightSource) {
         std::vector<std::string> line_tokens = {"AreaLightSource", R"("diffuse")", R"("color)", R"(L")", "[", "100.0", "100.0", "100.0", "]"};
         const std::unique_ptr<poly::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "AreaLightSource");
         EXPECT_EQ(directive->Identifier, "diffuse");
         EXPECT_EQ(directive->Arguments.size(), 1);
         EXPECT_EQ(directive->Arguments[0].Type, poly::PBRTArgument::PBRTArgumentType::pbrt_rgb);
         EXPECT_EQ(directive->Arguments[0].Name, "L");
         ASSERT_EQ(directive->Arguments[0].float_values->size(), 3);
         EXPECT_EQ(directive->Arguments[0].float_values->at(0), 100.f);
         EXPECT_EQ(directive->Arguments[0].float_values->at(1), 100.f);
         EXPECT_EQ(directive->Arguments[0].float_values->at(2), 100.f);
      }

      TEST(PBRTLex, TestBoolParamTrue) {
         std::vector<std::string> line_tokens = {"TestDirective", R"("identifier")", R"("bool)", R"(param_name")", R"("true")"};
         const std::unique_ptr<poly::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "TestDirective");
         EXPECT_EQ(directive->Identifier, "identifier");
         EXPECT_EQ(directive->Arguments.size(), 1);
         EXPECT_EQ(directive->Arguments[0].Type, poly::PBRTArgument::PBRTArgumentType::pbrt_bool);
         EXPECT_EQ(directive->Arguments[0].Name, "param_name");
         ASSERT_NE(directive->Arguments[0].bool_value, nullptr);
         EXPECT_TRUE(*(directive->Arguments[0].bool_value));
      }

      TEST(PBRTLex, TestBoolParamFalse) {
         std::vector<std::string> line_tokens = {"TestDirective", R"("identifier")", R"("bool)", R"(param_name")", R"("false")"};
         const std::unique_ptr<poly::PBRTDirective> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->Name, "TestDirective");
         EXPECT_EQ(directive->Identifier, "identifier");
         EXPECT_EQ(directive->Arguments.size(), 1);
         EXPECT_EQ(directive->Arguments[0].Type, poly::PBRTArgument::PBRTArgumentType::pbrt_bool);
         EXPECT_EQ(directive->Arguments[0].Name, "param_name");
         ASSERT_NE(directive->Arguments[0].bool_value, nullptr);
         EXPECT_FALSE(*(directive->Arguments[0].bool_value));
      }

      TEST(PBRTLex, TestBoolParamInvalid) {
         std::vector<std::string> line_tokens = {"TestDirective", R"("identifier")", R"("bool)", R"(param_name")", R"("asdf")"};
         ASSERT_THROW(test_lex(line_tokens), std::invalid_argument);
      }
   }
}