//
// Created by Daniel on 07-Apr-18.
//

#include "gtest/gtest.h"

#include "../../../src/common/parsers/pbrt_parser.h"
#include "../../../src/cpu/shapes/mesh.h"

namespace Tests {

   using poly::pbrt_parser;
   
   namespace Lex {
      
      std::unique_ptr<poly::pbrt_directive> test_lex(const std::vector<std::string> &line_tokens) {
         EXPECT_GT(line_tokens.size(), 0);
         return pbrt_parser::lex(line_tokens);
      }

      TEST(PBRTLex, DirectiveOnly) {
         std::vector<std::string> line_tokens = {"directive_name"};
         const std::unique_ptr<poly::pbrt_directive> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->identifier, "directive_name");
         EXPECT_EQ(directive->type, "");
         EXPECT_EQ(directive->arguments.size(), 0);
      }

      TEST(PBRTLex, DirectiveWithIdentifierWithoutArgument) {
         std::vector<std::string> line_tokens = {"directive_name", R"("identifier")"};
         const std::unique_ptr<poly::pbrt_directive> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->identifier, "directive_name");
         EXPECT_EQ(directive->type, "identifier");
         EXPECT_EQ(directive->arguments.size(), 0);
      }

      TEST(PBRTLex, ScaleDirective1) {
         std::vector<std::string> line_tokens = {"Scale", "1", "2", "3"};
         const std::unique_ptr<poly::pbrt_directive> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->identifier, "Scale");
         EXPECT_EQ(directive->type, "");
         EXPECT_EQ(directive->arguments.size(), 1);
         EXPECT_EQ(directive->arguments[0].Type, poly::pbrt_argument::pbrt_argument_type::pbrt_float);
         EXPECT_EQ(directive->arguments[0].Name, "");
         ASSERT_EQ(directive->arguments[0].float_values->size(), 3);
         EXPECT_EQ(directive->arguments[0].float_values->at(0), 1.f);
         EXPECT_EQ(directive->arguments[0].float_values->at(1), 2.f);
         EXPECT_EQ(directive->arguments[0].float_values->at(2), 3.f);
      }

      TEST(PBRTLex, ScaleDirective2) {
         std::vector<std::string> line_tokens = {"Scale", "1", "2"};
         ASSERT_EXIT( {const std::unique_ptr<poly::pbrt_directive> directive = test_lex(line_tokens); }, testing::ExitedWithCode(1), ".*");
      }

      TEST(PBRTLex, ScaleDirective3) {
         std::vector<std::string> line_tokens = {"Scale", "1", "2", "3", "4"};
         ASSERT_EXIT( {const std::unique_ptr<poly::pbrt_directive> directive = test_lex(line_tokens); }, testing::ExitedWithCode(1), ".*");
      }

      TEST(PBRTLex, AreaLightSource) {
         std::vector<std::string> line_tokens = {"AreaLightSource", R"("diffuse")", R"("color)", R"(L")", "[", "100.0", "100.0", "100.0", "]"};
         const std::unique_ptr<poly::pbrt_directive> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->identifier, "AreaLightSource");
         EXPECT_EQ(directive->type, "diffuse");
         EXPECT_EQ(directive->arguments.size(), 1);
         EXPECT_EQ(directive->arguments[0].Type, poly::pbrt_argument::pbrt_argument_type::pbrt_rgb);
         EXPECT_EQ(directive->arguments[0].Name, "L");
         ASSERT_EQ(directive->arguments[0].float_values->size(), 3);
         EXPECT_EQ(directive->arguments[0].float_values->at(0), 100.f);
         EXPECT_EQ(directive->arguments[0].float_values->at(1), 100.f);
         EXPECT_EQ(directive->arguments[0].float_values->at(2), 100.f);
      }

      TEST(PBRTLex, TestBoolParamTrue) {
         std::vector<std::string> line_tokens = {"TestDirective", R"("identifier")", R"("bool)", R"(param_name")", R"("true")"};
         const std::unique_ptr<poly::pbrt_directive> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->identifier, "TestDirective");
         EXPECT_EQ(directive->type, "identifier");
         EXPECT_EQ(directive->arguments.size(), 1);
         EXPECT_EQ(directive->arguments[0].Type, poly::pbrt_argument::pbrt_argument_type::pbrt_bool);
         EXPECT_EQ(directive->arguments[0].Name, "param_name");
         ASSERT_NE(directive->arguments[0].bool_value, nullptr);
         EXPECT_TRUE(*(directive->arguments[0].bool_value));
      }

      TEST(PBRTLex, TestBoolParamFalse) {
         std::vector<std::string> line_tokens = {"TestDirective", R"("identifier")", R"("bool)", R"(param_name")", R"("false")"};
         const std::unique_ptr<poly::pbrt_directive> directive = test_lex(line_tokens);
         ASSERT_NE(directive, nullptr);
         EXPECT_EQ(directive->identifier, "TestDirective");
         EXPECT_EQ(directive->type, "identifier");
         EXPECT_EQ(directive->arguments.size(), 1);
         EXPECT_EQ(directive->arguments[0].Type, poly::pbrt_argument::pbrt_argument_type::pbrt_bool);
         EXPECT_EQ(directive->arguments[0].Name, "param_name");
         ASSERT_NE(directive->arguments[0].bool_value, nullptr);
         EXPECT_FALSE(*(directive->arguments[0].bool_value));
      }

      TEST(PBRTLex, TestBoolParamInvalid) {
         std::vector<std::string> line_tokens = {"TestDirective", R"("identifier")", R"("bool)", R"(param_name")", R"("asdf")"};
         ASSERT_THROW(test_lex(line_tokens), std::invalid_argument);
      }
   }
}