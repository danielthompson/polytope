//
// Created by Daniel on 07-Apr-18.
//

#include <sstream>
#include <stack>
#include <cassert>
#include <unordered_map>
#include "pbrt_parser.h"
#include "mesh_parsers.h"
#include "../../cpu/integrators/PathTraceIntegrator.h"
#include "../../cpu/integrators/DebugIntegrator.h"
#include "../../cpu/cameras/PerspectiveCamera.h"
#include "../../cpu/samplers/samplers.h"
#include "../../cpu/runners/TileRunner.h"
#include "../../cpu/films/PNGFilm.h"
#include "../../cpu/filters/BoxFilter.h"
#include "../../cpu/scenes/Scene.h"
#include "../utilities/Common.h"
#include "../../cpu/scenes/skyboxes/ColorSkybox.h"
#include "../../cpu/structures/Vectors.h"
#include "../../cpu/shading/brdf/lambert_brdf.h"
#include "../../cpu/shading/brdf/mirror_brdf.h"
#include "../../cpu/shapes/mesh.h"
#include "../../cpu/shapes/tesselators.h"
#include "../../cpu/shading/brdf/glossy_brdf.h"

namespace poly {

   namespace {
      namespace str {
         
         const std::string AreaLightSource = "AreaLightSource";
         const std::string AttributeBegin = "AttributeBegin";
         const std::string AttributeEnd = "AttributeEnd";
         const std::string Camera = "Camera";
         const std::string filename = "filename";
         const std::string Film = "Film";
         const std::string halton = "halton";
         const std::string image = "image";
         const std::string Integrator = "Integrator";
         const std::string Kd = "Kd";
         const std::string Kr = "Kr";
         const std::string Ks = "Ks";
         const std::string LightSource = "LightSource";
         const std::string LookAt = "LookAt";
         const std::string MakeNamedMaterial = "MakeNamedMaterial";
         const std::string Material = "Material";
         const std::string matte = "matte";
         const std::string mirror = "mirror";
         const std::string name = "name";
         const std::string NamedMaterial = "NamedMaterial";
         const std::string ObjectBegin = "ObjectBegin";
         const std::string ObjectEnd = "ObjectEnd";
         const std::string ObjectInstance = "ObjectInstance";
         const std::string objmesh = "objmesh";
         const std::string PixelFilter = "PixelFilter";
         const std::string pixelsamples = "pixelsamples";
         const std::string plastic = "plastic";
         const std::string plymesh = "plymesh";
         const std::string Rotate = "Rotate";
         const std::string roughness = "roughness";
         const std::string Sampler = "Sampler";
         const std::string Scale = "Scale";
         const std::string Shape = "Shape";
         const std::string sphere = "sphere";
         const std::string Texture = "Texture";
         const std::string Transform = "Transform";
         const std::string TransformBegin = "TransformBegin";
         const std::string TransformEnd = "TransformEnd";
         const std::string Translate = "Translate";
         const std::string WorldBegin = "WorldBegin";
         const std::string WorldEnd = "WorldEnd";
      }
      
      std::string _inputFilename = "";
      std::string _basePathFromCWD = "";

      // strings

      enum DirectiveIdentifier {
         AreaLightSource,
         AttributeBegin,
         AttributeEnd,
         Camera,
         Film,
         Integrator,
         LightSource,
         LookAt,
         MakeNamedMaterial,
         Material,
         NamedMaterial,
         ObjectBegin,
         ObjectEnd,
         ObjectInstance,
         PixelFilter,
         Rotate,
         Sampler,
         Scale,
         Shape,
         Texture,
         Transform,
         TransformBegin,
         TransformEnd,
         Translate,
         WorldBegin,
         WorldEnd
      };

      const std::unordered_map<std::string, DirectiveIdentifier> SceneDirectiveMap {
            {str::Camera, Camera},
            {str::Film, Film},
            {str::Integrator, Integrator},
            {str::LookAt, LookAt},
            {str::PixelFilter, PixelFilter},
            {str::Rotate, Rotate},
            {str::Sampler, Sampler},
            {str::Scale, Scale},
            {str::Transform, Transform},
            {str::Translate, Translate},
      };

      const std::unordered_map<std::string, DirectiveIdentifier> WorldDirectiveMap {
            {str::AreaLightSource, AreaLightSource},
            {str::AttributeBegin, AttributeBegin},
            {str::AttributeEnd, AttributeEnd},
            {str::LightSource, LightSource},
            {str::MakeNamedMaterial, MakeNamedMaterial},
            {str::Material, Material},
            {str::NamedMaterial, NamedMaterial},
            {str::ObjectBegin, ObjectBegin},
            {str::ObjectEnd, ObjectEnd},
            {str::ObjectInstance, ObjectInstance},
            {str::Rotate, Rotate},
            {str::Scale, Scale},
            {str::Shape, Shape},
            {str::Texture, Texture},
            {str::Transform, Transform},
            {str::TransformBegin, TransformBegin},
            {str::TransformEnd, TransformEnd},
            {str::Translate, Translate},
            {str::WorldBegin, WorldBegin},
            {str::WorldEnd, WorldEnd},
      };

      enum ShapeIdentifier {
         objmesh,
         plymesh,
         sphere
      };

      const std::unordered_map<std::string, ShapeIdentifier> ShapeIdentifierMap {
            {str::objmesh, objmesh},
            {str::plymesh,    plymesh},
            {str::sphere,     sphere},
      };

      enum MaterialIdentifier {
         Matte,
         Mirror,
         Plastic
      };

      const std::unordered_map<std::string, MaterialIdentifier> MaterialIdentifierMap {
            {str::matte, Matte},
            { str::mirror, Mirror},
            { str::plastic, Plastic}
      };

      enum MaterialArgumentName {
         Kd,
         Ks,
         Roughness
      };

      const std::unordered_map<std::string, MaterialArgumentName> MaterialMatteArgumentMap {
            {str::Kd, Kd}
      };

      const std::unordered_map<std::string, MaterialArgumentName> MaterialPlasticArgumentMap {
            {str::Kd, Kd},
            {str::Ks, Ks},
            {str::roughness, Roughness}
      };
      
      enum OBJMeshArgument {
         Filename
      };

      const std::unordered_map<std::string, OBJMeshArgument> OBJMeshArgumentMap {
            {str::filename, Filename}
      };

      enum class PLYMeshArgument {
         Filename
      };

      const std::unordered_map<std::string, PLYMeshArgument> PLYMeshArgumentMap {
            {str::filename, PLYMeshArgument::Filename}
      };

      // TODO get rid of this junk in favor of using WorldDirectiveMap

      bool is_quoted(const std::string &token) {
         return (token[0] == '"' && token[token.size() - 1] == '"');
      }

      bool start_quoted(const std::string &token) {
         return (token[0] == '"' && token[token.size() - 1] != '"');
      }

      bool is_end_quoted(std::string token) {
         return (token[0] != '"' && token[token.size() - 1] == '"');
      }

      void LogOther(const std::unique_ptr<pbrt_directive> &directive, const std::string &error) {
         Log.warning(directive->identifier + ": " + error);
      }

      void log_illegal_directive(const std::unique_ptr<pbrt_directive> &directive, const std::string &error) {
         Log.error(directive->identifier + ": " + error);
      }

      void log_illegal_identifier(const std::unique_ptr<pbrt_directive> &directive, const std::string &error) {
         Log.error(directive->identifier + ": \"" + directive->type + "\": " + error);
      }

      void LogMissingArgument(const std::unique_ptr<pbrt_directive>& directive, const std::string& argument) {
         Log.warning("Directive [" + directive->identifier + "] w/ identifier [" + directive->type + "] is missing argument [" + argument + "]");
      }

      void LogMissingDirective(const std::string& name, std::string& defaultOption) {
         Log.warning("Directive [" + name + "] is missing, defaulting to " + defaultOption + ".");
      }

      void LogUnknownDirective(const std::unique_ptr<pbrt_directive> &directive) {
         Log.warning("Directive [" + directive->identifier + "] found, but is unknown. Ignoring.");
      }

      void LogUnknownIdentifier(const std::unique_ptr<pbrt_directive> &directive) {
         Log.warning("Directive [" + directive->identifier + "] has unknown identifier [" + directive->type + "].");
      }

      void LogUnknownArgument(const pbrt_argument &argument) {
         
         Log.warning("Unknown argument type/name combination: [" +
                     pbrt_argument::get_argument_type_string(argument.Type) + "] / [" + argument.Name + "].");
      }

      void LogWrongArgumentType(const std::unique_ptr<pbrt_directive> &directive, const pbrt_argument &argument) {
         Log.warning("Directive [" + directive->identifier + "] w/ identifier [" + directive->type + "] found has argument [" + argument.Name + "] with wrong type [" +
                     pbrt_argument::get_argument_type_string(argument.Type) + "].");
      }

      void LogUnimplementedDirective(const std::unique_ptr<pbrt_directive> &directive) {
         Log.warning("Directive [" + directive->identifier + "] w/ identifier [" + directive->type + "] found, but is not yet implemented. Ignoring.");
      }

      unsigned int stoui(const std::string& text) {
         return static_cast<unsigned int>(stoi(text));
      }

      // defaults
      constexpr float DefaultCameraFOV = 90.f;
      constexpr unsigned int DefaultBoundsX = 640;
      constexpr unsigned int DefaultBoundsY = 360;
      constexpr unsigned int DefaultSamples = 8;
   }

   std::unique_ptr<AbstractRunner> pbrt_parser::parse_file(const std::string &filepath) {

      std::unique_ptr<std::vector<std::vector<std::string>>> tokens = scan(open_ascii_stream(filepath));

      // determine the name of the file from the given path
      const size_t lastPos = filepath.find_last_of(UnixPathSeparator);

      if (lastPos == std::string::npos) {
         _inputFilename = filepath;
         _basePathFromCWD = "";
      }
      else {
         _basePathFromCWD = filepath.substr(0, lastPos + 1);
         _inputFilename = filepath.substr(lastPos + 1);
      }

      return parse(std::move(tokens));
   }

   std::unique_ptr<AbstractRunner> pbrt_parser::parse_string(const std::string &text) {
      auto tokens = scan(std::make_unique<std::istringstream>(text));
      return parse(std::move(tokens));
   }

   std::unique_ptr<std::vector<std::vector<std::string>>> pbrt_parser::scan(const std::unique_ptr<std::istream> &stream) {

      std::unique_ptr<std::vector<std::vector<std::string>>> tokens = std::make_unique<std::vector<std::vector<std::string>>>();

      int sourceLineNumber = 0;
      int targetLineNumber = -1;
      std::string line;
      while (getline(*stream, line)) {
         if (line.empty()) {
            continue;
         }
         
         tokens->emplace_back();
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         while (iss >> word) {
            // strip out comments
            if (word.find('#') == 0)
               break;

            // check if this is a directive
            if (WorldDirectiveMap.count(word) > 0 || SceneDirectiveMap.count(word) > 0) {
               // if this is a directive, then we move on to a new line
               targetLineNumber++;
            }

            // split brackets, if needed

            if (word.empty()) {
               continue;
            }

            const unsigned long lastIndex = word.size() - 1;

            const bool first_char_is_bracket = word[0] == '[';
            const bool last_char_is_bracket = word[lastIndex] == ']';
            const unsigned int token_index = (*tokens)[targetLineNumber].size() - 1;

            std::vector<std::string>* current_line = &(*tokens)[targetLineNumber];
            
            if (word.size() == 1) {
               if (last_char_is_bracket) {
                  if ((*current_line)[token_index - 1] == "[") {
                     current_line->erase(current_line->end() - 2);
                  }
                  else {
                     current_line->push_back("]");
                  }
               }
               else {
                  current_line->push_back(word);
               }
            }
            else {
               if (first_char_is_bracket) {
                  if (last_char_is_bracket) {
                     current_line->push_back(word.substr(1, word.size() - 2));
                  }
                  else {
                     current_line->push_back("[");
                     current_line->push_back(word.substr(1));
                  }
               }
               else {
                  if (last_char_is_bracket) {
                     if ((*current_line)[token_index] == "[") {
                        current_line->erase(current_line->end() - 1);
                        current_line->push_back(word.substr(0, word.size() - 1));
                     }
                     else {
                        current_line->push_back(word.substr(0, word.size() - 1));
                        current_line->push_back("]");
                     }
                  }
                  else {
                     current_line->push_back(word);
                  }
               }
            }
            assert(!current_line->empty());
         }
         
         // just-added line could still be empty, like if it was non-empty but had no valid tokens
         if (tokens->back().empty()) {
            tokens->erase(tokens->end() - 1);
         }
         
         sourceLineNumber++;
      } // end line
      
      Log.debug("Scan complete.");
      return tokens;
   }

   void LexFloatArrayArgument(const std::vector<std::string>& line, const int expected_num_elements, const bool in_brackets, poly::pbrt_argument *argument) {
      const int expected_num_tokens = expected_num_elements + 1 + (in_brackets ? 2 : 0);
      
      if (line.size() != expected_num_tokens) {
         ERROR("LexFloatArrayArgument(): " + line[0] + " requires exactly " + std::to_string(expected_num_elements) + " arguments, but found " + std::to_string(line.size()));
      }

      if (in_brackets) {
         if (line[1] != "[") {
            ERROR("LexFloatArrayArgument(): Expected an opening bracket for token 1, but found ["  + line[1] + "]");
         }
         const int last_index = line.size() - 1;
         if (line[last_index] != "]") {
            ERROR("LexFloatArrayArgument(): Expected a closet bracket for token " + std::to_string(last_index) + ", but found ["  + line[last_index] + "]");
         }
      }
      
      argument->Type = pbrt_argument::pbrt_float;
      argument->float_values = std::make_unique<std::vector<float>>();
      
      const int starting_index = in_brackets ? 2 : 1;
      
      for (int i = starting_index; i <= expected_num_elements; i++) {
         std::string current_arg = line[i];
         
         float value;
         try {
            value = stof(current_arg);
         }
         catch (const std::invalid_argument &) {
            ERROR("LexFloatArrayArgument():" + line[0] + ": failed to parse [" + current_arg + "] as a float");
         }
         argument->float_values->push_back(value);
      }
   }
   
   std::unique_ptr<poly::pbrt_directive> pbrt_parser::lex(std::vector<std::string> line) {
      std::unique_ptr<poly::pbrt_directive> directive = std::make_unique<pbrt_directive>();

      if (line.empty()) {
         Log.warning("lex(): empty line. Hmm...");
         return directive;
      }
      
      Log.debug("lex(): processing line starting with [" + line[0] + "].");
      
      bool debug = false;
      if (line[0] == "Texture")
         bool debug = true;

      directive->identifier = line[0];
      
      if (line.size() == 1) {
         Log.debug("lex(): directive [" + directive->identifier + "] OK");
         return directive;
      }

      int argument_start_index = 2;
      
      // TODO put this into a switch
      
      // first, lex directives that use non-uniform argument syntax
      if (directive->identifier == str::LookAt) {
         directive->arguments = std::vector<pbrt_argument>();
         directive->arguments.emplace_back(poly::pbrt_argument::pbrt_argument_type::pbrt_float);
         LexFloatArrayArgument(line, 9, false, &(directive->arguments[0]));
         Log.debug("lex(): directive [" + directive->identifier + "] OK");
         return directive;
      }
      
      else if (directive->identifier == str::Rotate) {
         directive->arguments = std::vector<pbrt_argument>();
         directive->arguments.emplace_back(poly::pbrt_argument::pbrt_argument_type::pbrt_float);
         LexFloatArrayArgument(line, 4, false, &(directive->arguments[0]));
         Log.debug("lex(): directive [" + directive->identifier + "] OK");
         return directive;
      }

      else if (directive->identifier == str::Scale) {
         directive->arguments = std::vector<pbrt_argument>();
         directive->arguments.emplace_back(poly::pbrt_argument::pbrt_argument_type::pbrt_float);
         LexFloatArrayArgument(line, 3, false, &(directive->arguments[0]));
         Log.debug("lex(): directive [" + directive->identifier + "] OK");
         return directive;
      }

      else if (directive->identifier == str::Texture) {
         // name
         std::string name = line[1].substr(1, line[1].length() - 2);
         if (name.empty()) {
            ERROR("lex(): directive [" + directive->identifier + "] cannot have an empty name");
         }
         directive->name = name;
         
         // type
         std::string type = line[2].substr(1, line[2].length() - 2);
         if (type.empty()) {
            ERROR("lex(): directive [" + directive->identifier + "] cannot have an empty type");
         }
         directive->type = type;
         
         // class
         std::string class_name = line[3].substr(1, line[3].length() - 2);
         if (class_name.empty()) {
            ERROR("lex(): directive [" + directive->identifier + "] cannot have an empty class name");
         }
         directive->class_name = class_name;
         
         // regular arguments
         argument_start_index = 4;
         
         Log.debug("lex(): directive [" + directive->identifier + "] started OK");
      }

      else if (directive->identifier == str::Transform) {
         directive->arguments = std::vector<pbrt_argument>();
         directive->arguments.emplace_back(poly::pbrt_argument::pbrt_argument_type::pbrt_float);
         LexFloatArrayArgument(line,  16, true, &(directive->arguments[0]));
         Log.debug("lex(): directive [" + directive->identifier + "] OK");
         return directive;
      }
      
      else if (directive->identifier == str::Translate) {
         directive->arguments = std::vector<pbrt_argument>();
         directive->arguments.emplace_back(poly::pbrt_argument::pbrt_argument_type::pbrt_float);
         LexFloatArrayArgument(line, 3, false, &(directive->arguments[0]));
         Log.debug("lex(): directive [" + directive->identifier + "] OK");
         return directive;
      }
      
      else {
         if (is_quoted(line[1])) {
            directive->type = line[1].substr(1, line[1].length() - 2);
         }
         else {
            ERROR("lex(): second token (identifier) %s isn't quoted, but should be", line[0].c_str());
         }
      } 


      if (line.size() == 2) {
         Log.debug("Lexed directive [" + directive->identifier + "]");
         return directive;
      }
      
      bool inValue = false;
      bool in_arg = false;
      int i = argument_start_index;
      int current_arg_index = -1;
      pbrt_argument* current_arg = nullptr;
      while (i < line.size()) {
         if (start_quoted(line[i]) && is_end_quoted(line[i + 1])) {
            // we're in an argument
            if (directive->arguments.empty())
               directive->arguments = std::vector<pbrt_argument>();
            const pbrt_argument::pbrt_argument_type type = pbrt_argument::get_argument_type(line[i].substr(1, line[i].length() - 1));
            directive->arguments.emplace_back(pbrt_argument(type));
            current_arg_index++;
            current_arg = &(directive->arguments[current_arg_index]);
            current_arg->Name = line[i + 1].substr(0, line[i + 1].length() - 1);
            inValue = true;
            i += 2;
            continue;
         }
         // TODO catch sequential non-bracketed values
         if (line[i] == "[") {
            inValue = true;
            i++;
            continue;
         }
         if (line[i] == "]") {
            inValue = false;
            i++;
            continue;
         }
         if (inValue) {
            if (is_quoted(line[i])) {
               // probably just string?
               assert(current_arg->int_values == nullptr);
               assert(current_arg->float_values == nullptr);
               
               std::string value = line[i].substr(1, line[i].length() - 2);
               
               switch (current_arg->Type) {
                  case pbrt_argument::pbrt_argument_type::pbrt_texture:
                  case pbrt_argument::pbrt_argument_type::pbrt_string: {
                     assert(current_arg->bool_value == nullptr);
                     current_arg->string_value = std::make_unique<std::string>(value);
                     break;
                  }
                  case pbrt_argument::pbrt_argument_type::pbrt_bool: {
                     assert(current_arg->string_value == nullptr);
                     if (value == "true") {
                        current_arg->bool_value = std::make_unique<bool>(true);
                     }
                     else if (value == "false") {
                        current_arg->bool_value = std::make_unique<bool>(false);
                     }
                     else {
                        throw std::invalid_argument(line[0] + ": failed to parse [" + value + "] as a bool");
                     }
                     break;
                  }
                  default: {
                     // unimplemented arg type
                     assert(false);
                  }
               }
            } 
            else {
               switch (current_arg->Type) {
                  case pbrt_argument::pbrt_rgb:
                  case pbrt_argument::pbrt_float: {
                     assert(current_arg->string_value == nullptr);
                     assert(current_arg->int_values == nullptr);
                     
                     float value;
                     try {
                        value = std::stof(line[i]);
                     }
                     catch (const std::invalid_argument &) {
                        throw std::invalid_argument(line[0] + ": failed to parse [" + line[i] + "] as a float");
                     }
//                     if (current_arg->Type == PBRTArgument::pbrt_rgb) {
//                        if (value < 0.f || value > 1.f) {
//                           throw std::invalid_argument(line[0] + ": parsed value [" + std::to_string(value) + "] is outside the range for rgb (must be between 0 and 1 inclusive)");
//                        }
//                     }
                     if (current_arg->float_values == nullptr) {
                        current_arg->float_values = std::make_unique<std::vector<float>>();
                     }

                     current_arg->float_values->push_back(value);
                     break;
                  }
                  case pbrt_argument::pbrt_int: {
                     assert(current_arg->string_value == nullptr);
                     assert(current_arg->float_values == nullptr);
                     int value;
                     try {
                        value = std::stoi(line[i]);
                     }
                     catch (const std::invalid_argument &) {
                        throw std::invalid_argument(line[0] + ": failed to parse [" + line[i] + "] as an int");
                     }

                     if (current_arg->int_values == nullptr) {
                        current_arg->int_values = std::make_unique<std::vector<int>>();
                     }
                     
                     current_arg->int_values->push_back(value);
                     break;
                  }
                  default: {
                     // TODO
                  }
               }
            }
            i++;
            continue;
         }
         ERROR("lex(): current line has invalid token [" + line[i] + "] :(");
      }

//      if (inValue) {
//         directive->Arguments.push_back(argument);
//      }

      Log.debug("lex(): directive [" + directive->identifier + "] OK");

      return directive;
   }
   
   static std::unique_ptr<poly::Transform> rotate_directive(const std::unique_ptr<poly::pbrt_directive>& directive) {
      // TODO need to ensure just 1 argument with 4 values
      
      const float angle = directive->arguments[0].float_values->at(0) * PIOver180;
      float x = directive->arguments[0].float_values->at(1);
      float y = directive->arguments[0].float_values->at(2);
      float z = directive->arguments[0].float_values->at(3);

      // normalize
      const float length_inverse = 1.f / std::sqrt(x * x + y * y + z * z);
      x *= length_inverse;
      y *= length_inverse;
      z *= length_inverse;
      
      return std::make_unique<poly::Transform>(Transform::Rotate(angle, x, y, z));
   }
   
   static std::unique_ptr<poly::Transform> scale_directive(const std::unique_ptr<poly::pbrt_directive>& directive) {
      pbrt_argument* arg = &(directive->arguments[0]);
      float x = arg->float_values->at(0);
      float y = arg->float_values->at(1);
      float z = arg->float_values->at(2);

      return std::make_unique<poly::Transform>(Transform::Scale(x, y, z));
   }
   
   static std::unique_ptr<poly::Transform> transform_directive(const std::unique_ptr<poly::pbrt_directive>& directive) {
      pbrt_argument* arg = &(directive->arguments[0]);
      float m00 = arg->float_values->at(0);
      float m01 = arg->float_values->at(1);
      float m02 = arg->float_values->at(2);
      float m03 = arg->float_values->at(3);

      float m10 = arg->float_values->at(4);
      float m11 = arg->float_values->at(5);
      float m12 = arg->float_values->at(6);
      float m13 = arg->float_values->at(7);

      float m20 = arg->float_values->at(8);
      float m21 = arg->float_values->at(9);
      float m22 = arg->float_values->at(10);
      float m23 = arg->float_values->at(11);

      float m30 = arg->float_values->at(12);
      float m31 = arg->float_values->at(13);
      float m32 = arg->float_values->at(14);
      float m33 = arg->float_values->at(15);

      return std::make_unique<poly::Transform>(m00, m01, m02, m03,
                                               m10, m11, m12, m13,
                                               m20, m21, m22, m23,
                                               m30, m31, m32, m33);
   }
   
   static std::unique_ptr<poly::Transform> translate_directive(const std::unique_ptr<poly::pbrt_directive>& directive) {
      // need to ensure just one argument with 3 values
      pbrt_argument* arg = &(directive->arguments[0]);
      float x = arg->float_values->at(0);
      float y = arg->float_values->at(1);
      float z = arg->float_values->at(2);
      return std::make_unique<poly::Transform>(Transform::Translate(x, y, z));
   }
   
   std::unique_ptr<AbstractRunner> pbrt_parser::parse(std::unique_ptr<std::vector<std::vector<std::string>>> tokens) noexcept(false){
      std::vector<std::unique_ptr<pbrt_directive>> scene_directives;
      std::vector<std::unique_ptr<pbrt_directive>> world_directives;

      {
         std::vector<std::unique_ptr<pbrt_directive>>* current_directives = &scene_directives;
         for (const std::vector<std::string>& line : *tokens) {
            std::unique_ptr<pbrt_directive> directive = lex(line);
            // TODO ensure directive is valid for scene/world
            
            if (directive->identifier == str::WorldBegin) {
               current_directives = &world_directives;
            }
            
            current_directives->push_back(std::move(directive));
         }
      }

      // sampler

      bool missingSampler = true;

      unsigned int numSamples = DefaultSamples;

      for (const std::unique_ptr<pbrt_directive> &directive : scene_directives) {
         if (directive->identifier == str::Sampler) {
            missingSampler = false;
            if (directive->type == str::halton) {
               sampler = std::make_unique<HaltonSampler>();
            } else {
               LogUnknownIdentifier(directive);
               sampler = std::make_unique<HaltonSampler>();
            }

            for (const pbrt_argument& arg : directive->arguments) {
               if (arg.Type == pbrt_argument::pbrt_int) {
                  if (arg.Name == str::pixelsamples) {
                     numSamples = arg.int_values->at(0);
                     break;
                  } else {
                     LogUnknownArgument(arg);
                  }
                  break;
               }
               else {
                  // TODO log bad argument type
               }
            }
            break;
         }
      }

      if (missingSampler) {
         std::string defaultOption = "Halton";
         LogMissingDirective(str::Sampler, defaultOption);
      }

      if (sampler == nullptr) {
         sampler = std::make_unique<CenterSampler>();
      }

      Log.debug("Made (center) sampler.");

      // film

      bool missingFilm = true;

      for (const std::unique_ptr<pbrt_directive> &directive : scene_directives) {
         if (directive->identifier == str::Film) {
            if (directive->type == str::image) {
               unsigned int x = 0;
               unsigned int y = 0;

               const auto dotIndex = _inputFilename.find_last_of('.');
               std::string filename;
               if (dotIndex == std::string::npos)
                  filename = _inputFilename;
               else
                  filename = _inputFilename.substr(0, dotIndex) + ".png";

               bool foundX = false;
               bool foundY = false;

               for (const pbrt_argument& arg : directive->arguments) {
                  if (arg.Type == pbrt_argument::pbrt_int) {
                     if (arg.Name == "xresolution") {
                        x = arg.int_values->at(0);
                        foundX = true;
                     } else if (arg.Name == "yresolution") {
                        y = arg.int_values->at(0);
                        foundY = true;
                     } else {
                        LogUnknownArgument(arg);
                     }
                  } else if (arg.Type == pbrt_argument::pbrt_string) {
                     if (arg.Name == "filename") {
                        filename = *(arg.string_value);
                        // TODO complain if filename arg is provided but empty
                     } else {
                        LogUnknownArgument(arg);
                     }
                  }
               }

               bounds.x = foundX ? x : DefaultBoundsX;
               bounds.y = foundY ? y : DefaultBoundsY;

               missingFilm = false;

               film = std::make_unique<PNGFilm>(bounds, filename, std::move(filter));
            }
            break;
         }
      }

      if (missingFilm) {
         std::string defaultOption = "PNGFilm with 640x480 and output filename polytope.png";
         LogMissingDirective(str::Film, defaultOption);
      }

      if (film == nullptr) {
         std::string filename = "polytope.png";
         bounds.x = DefaultBoundsX;
         bounds.y = DefaultBoundsY;
         film = std::make_unique<PNGFilm>(bounds, filename, std::move(filter));
      }

      Log.debug("Made film.");

      // filter

      bool missingFilter = true;

      for (const std::unique_ptr<pbrt_directive>& directive : scene_directives) {
         if (directive->identifier == str::PixelFilter) {
            missingFilter = false;
            if (directive->type == "box") {
               unsigned int xWidth = 0;
               unsigned int yWidth = 0;

               for (const pbrt_argument& arg : directive->arguments) {
                  if (arg.Type == pbrt_argument::pbrt_int) {
                     if (arg.Name == "xwidth") {
                        xWidth = arg.int_values->at(0);
                     } else if (arg.Name == "ywidth") {
                        yWidth = arg.int_values->at(0);
                     } else {
                        LogUnknownArgument(arg);
                     }
                     break;
                  }
               }
            } else {
               LogUnknownIdentifier(directive);
            }
            break;
         }
      }

      if (missingFilter) {
         std::string defaultOption = "Box";
         LogMissingDirective(str::PixelFilter, defaultOption);
      }

      if (filter == nullptr) {
         filter = std::make_unique<BoxFilter>(bounds);
         film->Filter = std::move(filter);
      }

      Log.debug("Made filter.");

      // camera

      std::unique_ptr<AbstractCamera> camera;

      {
         Point eye;
         Point lookAt;
         Vector up;

         poly::Transform current_transform;
         
         for (const std::unique_ptr<pbrt_directive>& directive : scene_directives) {
            
            DirectiveIdentifier identifier;
            try {
               identifier = SceneDirectiveMap.at(directive->identifier);
            }
            catch (...) {
               ERROR("Indentifier [%s] is not valid in the scene block");
            }
            switch (identifier) {
               case DirectiveIdentifier::LookAt: {
                  const float eyeX = directive->arguments[0].float_values->at(0);
                  const float eyeY = directive->arguments[0].float_values->at(1);
                  const float eyeZ = directive->arguments[0].float_values->at(2);

                  eye = Point(eyeX, eyeY, eyeZ);

                  const float lookAtX = directive->arguments[0].float_values->at(3);
                  const float lookAtY = directive->arguments[0].float_values->at(4);
                  const float lookAtZ = directive->arguments[0].float_values->at(5);

                  lookAt = Point(lookAtX, lookAtY, lookAtZ);

                  const float upX = directive->arguments[0].float_values->at(6);
                  const float upY = directive->arguments[0].float_values->at(7);
                  const float upZ = directive->arguments[0].float_values->at(8);

                  up = Vector(upX, upY, upZ);

                  poly::Transform t = Transform::LookAt(eye, lookAt, up, false);
                  current_transform *= t;

                  Log.debug("Found LookAt.");
                  break;
               }
               case DirectiveIdentifier::Rotate: {
                  current_transform *= *rotate_directive(directive);
                  break;
               }
               case DirectiveIdentifier::Scale: {
                  current_transform *= *scale_directive(directive);
                  break;
               }
               case DirectiveIdentifier::Transform: {
                  current_transform *= *transform_directive(directive);
                  break;
               }
               case DirectiveIdentifier::Translate: {
                  current_transform *= *translate_directive(directive);
                  break;
               }
               default:
                  // intentionally take no action if it's not a transformation directive
                  break;
            }
         }

         CameraSettings settings = CameraSettings(bounds, DefaultCameraFOV);

         bool foundCamera = false;

         for (const std::unique_ptr<pbrt_directive> &directive : scene_directives) {
            if (directive->identifier == str::Camera) {
               if (directive->type == "perspective") {
                  float fov = DefaultCameraFOV;

                  foundCamera = true;

                  for (const pbrt_argument& arg : directive->arguments) {
                     if (arg.Type == pbrt_argument::pbrt_float) {
                        if (arg.Name == "fov") {
                           fov = arg.float_values->at(0);
                        } 
                        else {
                           LogUnknownArgument(arg);
                        }
                     }
                  }

                  settings.FieldOfView = fov;

                  camera = std::make_unique<PerspectiveCamera>(settings, current_transform, true);
                  camera->eye = eye;
                  camera->lookAt = lookAt;
                  camera->up = up;
                  break;
               }
            }
         }

         if (!foundCamera) {
            std::ostringstream stringstream;
            stringstream << "PerspectiveCamera with FOV = " << DefaultCameraFOV;
            std::string cameraDefaultString = stringstream.str();
            LogMissingDirective(str::Camera, cameraDefaultString);
            camera = std::make_unique<PerspectiveCamera>(settings, current_transform, true);
         }
      }

      // integrator

      bool missingIntegrator = true;

      for (const std::unique_ptr<pbrt_directive>& directive : scene_directives) {
         if (directive->identifier == str::Integrator) {
            if (directive->type == "path") {
               unsigned int maxDepth = 5;

               missingIntegrator = false;

               bool missingDepth = true;

               for (const pbrt_argument& arg : directive->arguments) {
                  if (arg.Type == pbrt_argument::pbrt_int) {
                     if (arg.Name == "maxdepth") {
                        maxDepth = arg.int_values->at(0);
                        missingDepth = false;
                        break;
                     } else {
                        LogUnknownArgument(arg);
                     }
                  }
               }

               if (missingDepth) {
                  LogMissingArgument(directive, "maxdepth");
               }
               //_integrator = std::make_unique<poly::DebugIntegrator>(maxDepth);
               integrator = std::make_unique<poly::PathTraceIntegrator>(maxDepth);
            }
            else {
               LogUnimplementedDirective(directive);
            }
            break;
         }
      }

      if (missingIntegrator) {
         std::string error = "PathTraceIntegrator with MaxDepth = 5";
         LogMissingDirective(str::Integrator, error);
      }

      if (integrator == nullptr) {
         integrator = std::make_unique<PathTraceIntegrator>(5);
      }

      // world

      std::vector<std::shared_ptr<poly::Material>> namedMaterials;

      std::stack<std::shared_ptr<poly::Material>> materialStack;
      std::stack<std::shared_ptr<SpectralPowerDistribution>> lightStack;
      std::stack<std::shared_ptr<poly::Transform>> transformStack;

      std::shared_ptr<poly::Material> activeMaterial;
      std::shared_ptr<SpectralPowerDistribution> activeLight;
      std::shared_ptr<poly::Transform> activeTransform = std::make_shared<poly::Transform>();
      
      /**
       * Maps mesh names to previously-defined mesh geometries.
       */
      std::unordered_map<std::string, std::shared_ptr<poly::mesh_geometry>> name_mesh_map;
      std::shared_ptr<poly::mesh_geometry> current_geometry = nullptr;

      scene = new Scene(std::move(camera));

      bool in_object_begin = false;
      
      for (const std::unique_ptr<pbrt_directive>& directive : world_directives) {
         DirectiveIdentifier name;
         try {
            name = WorldDirectiveMap.at(directive->identifier);
         }
         catch (...) {
            LogUnknownDirective(directive);
            continue;
         }

         switch (name) {
            case DirectiveIdentifier::AreaLightSource: {
               // lights with geometry
               if (directive->type != "diffuse") {
                  LogUnknownIdentifier(directive);
                  break;
               }
               for (const pbrt_argument& argument : directive->arguments) {
                  if (argument.Name == "L") {
                     if (activeLight == nullptr) {
                        activeLight = std::make_shared<SpectralPowerDistribution>();
                     }
                     activeLight->r = argument.float_values->at(0) * 255;
                     activeLight->g = argument.float_values->at(1) * 255;
                     activeLight->b = argument.float_values->at(2) * 255;
                     break;
                  }
               }
               break;
            }
            case DirectiveIdentifier::AttributeBegin: {
               // push onto material stack
               if (activeMaterial != nullptr) {
                  materialStack.push(activeMaterial);
               }

               // push onto light stack
               if (activeLight != nullptr) {
                  lightStack.push(activeLight);
               }

               // push onto transform stack
               transformStack.push(activeTransform);
               activeTransform = std::make_shared<poly::Transform>(*(activeTransform.get()));
               break;
            }
            case DirectiveIdentifier::AttributeEnd: {
               // pop material stack
               if (!materialStack.empty()) {
                  std::shared_ptr<poly::Material> stackValue = materialStack.top();
                  materialStack.pop();
                  activeMaterial = stackValue;
               }

               // pop light stack
               if (lightStack.empty()) {
                  activeLight = nullptr;
               } else {
                  std::shared_ptr<SpectralPowerDistribution> stackValue = lightStack.top();
                  lightStack.pop();
                  activeLight = stackValue;
               }

               // pop transform stack
               if (!transformStack.empty()) {
                  std::shared_ptr<poly::Transform> stackValue = transformStack.top();
                  transformStack.pop();
                  assert (stackValue != nullptr);
                  activeTransform = stackValue;
               }
               break;
            }
            case DirectiveIdentifier::LightSource: {
               // lights without geometry
               if (directive->type == "infinite") {
                  for (const pbrt_argument& argument : directive->arguments) {
                     if (argument.Name == "L") {
                        const float r = argument.float_values->at(0);
                        const float g = argument.float_values->at(1);
                        const float b = argument.float_values->at(2);

                        const poly::SpectralPowerDistribution spd(r * 255, g * 255, b * 255);

                        scene->Skybox = std::make_unique<ColorSkybox>(spd);
                        break;
                     }
                  }
               }
               else {
                  LogUnimplementedDirective(directive);
               }
               break;
            }
            case DirectiveIdentifier::MakeNamedMaterial: {
               const std::string materialName = directive->type;
               std::shared_ptr<AbstractBRDF> brdf;
               poly::ReflectanceSpectrum reflectanceSpectrum;
               for (const pbrt_argument &argument : directive->arguments) {
                  if (argument.Type == pbrt_argument::pbrt_string) {
                     if (argument.Name == "type" && *(argument.string_value) == str::matte) {
                        // TODO instead of using default, parse values
                        ReflectanceSpectrum refl(0.5f, 0.5f, 0.5f);
                        brdf = std::make_unique<poly::LambertBRDF>(refl);
                     }
                  }
                  if (argument.Type == pbrt_argument::pbrt_rgb) {
                     if (argument.Name == str::Kd) {
                        reflectanceSpectrum.r = argument.float_values->at(0);
                        reflectanceSpectrum.g = argument.float_values->at(1);
                        reflectanceSpectrum.b = argument.float_values->at(2);
                     }
                  }
               }

               // TODO use hash table instead
               namedMaterials.push_back(std::make_shared<poly::Material>(std::move(brdf), reflectanceSpectrum, materialName));
               break;
            }
            case DirectiveIdentifier::Material: {
               MaterialIdentifier identifier;
               try {
                  identifier = MaterialIdentifierMap.at(directive->type);
               }
               catch (...) {
                  LogUnknownIdentifier(directive);
                  continue;
               }
               switch (identifier) {
                  case MaterialIdentifier::Plastic: {
                     // diffuse reflectivity
                     ReflectanceSpectrum kd(0.25f, 0.25f, 0.25f);

                     // specular reflectivity
                     ReflectanceSpectrum ks(0.25f, 0.25f, 0.25f);

                     float roughness = 0.1f;

                     for (const pbrt_argument &argument : directive->arguments) {
                        MaterialArgumentName param;
                        try {
                           param = MaterialPlasticArgumentMap.at(argument.Name);
                        }
                        catch (...) {
                           LogUnknownArgument(argument);
                           continue;
                        }
                        switch (param) {
                           case Kd: {
                              if (argument.Type == pbrt_argument::pbrt_rgb) {
                                 kd.r = argument.float_values->at(0);
                                 kd.g = argument.float_values->at(1);
                                 kd.b = argument.float_values->at(2);
                              }
                              break;
                           }
                           case Ks: {
                              if (argument.Type == pbrt_argument::pbrt_rgb) {
                                 ks.r = argument.float_values->at(0);
                                 ks.g = argument.float_values->at(1);
                                 ks.b = argument.float_values->at(2);
                              }
                              break;
                           }
                           case Roughness: {
                              if (argument.Type == pbrt_argument::pbrt_float) {
                                 roughness = argument.float_values->at(0);
                              }
                              break;
                           }
                           default: {
                              LogUnknownArgument(argument);
                              break;
                           }
                        }
                     }
                     
                     std::shared_ptr<poly::AbstractBRDF> brdf = std::make_shared<poly::GlossyBRDF>(ks, kd, roughness);
                     std::shared_ptr<poly::Material> material = std::make_shared<poly::Material>(brdf);
                     activeMaterial = material;
                     break;
                  }
                  case MaterialIdentifier::Matte: {
                     for (const pbrt_argument& argument : directive->arguments) {
                        MaterialArgumentName param;
                        try {
                           param = MaterialMatteArgumentMap.at(argument.Name);
                        }
                        catch (...) {
                           LogUnknownArgument(argument);
                           continue;
                        }
                        switch (param) {
                           case MaterialArgumentName::Kd: {
                              
                           }
                        }
                        if (argument.Name == str::Kd && argument.Type == pbrt_argument::pbrt_rgb) {
                           const float r = argument.float_values->at(0);
                           const float g = argument.float_values->at(1);
                           const float b = argument.float_values->at(2);

                           ReflectanceSpectrum refl(r, g, b);
                           std::shared_ptr<poly::AbstractBRDF> brdf = std::make_shared<poly::LambertBRDF>(refl);
                           std::shared_ptr<poly::Material> material = std::make_shared<poly::Material>(brdf);
                           activeMaterial = material;
                        }
                     }
                     break;
                  }
                  case MaterialIdentifier::Mirror: {
                     for (const pbrt_argument& argument : directive->arguments) {
                        if (argument.Name == str::Kr && argument.Type == pbrt_argument::pbrt_rgb) {
                           const float r = argument.float_values->at(0);
                           const float g = argument.float_values->at(1);
                           const float b = argument.float_values->at(2);

                           ReflectanceSpectrum refl(r, g, b);
                           std::shared_ptr<poly::AbstractBRDF> brdf = std::make_shared<poly::MirrorBRDF>(refl);
                           std::shared_ptr<poly::Material> material = std::make_shared<poly::Material>(brdf);
                           activeMaterial = material;
                        }
                     }
                     break;
                  }
               }
               break;
            }
            case DirectiveIdentifier::NamedMaterial: {
               std::string materialName = directive->type;
               bool found = false;
               // TODO hash table
               for (const auto &material : namedMaterials) {
                  if (material->Name == materialName) {
                     activeMaterial = material;
                     found = true;
                     break;
                  }
               }
               if (!found) {
                  LogOther(directive, "Specified material [" + materialName + "] not found. Have you defined it yet?");
               }
               break;
            }
            case DirectiveIdentifier::ObjectBegin: {
               if (in_object_begin) {
                  log_illegal_directive(directive, "ObjectBegin directive cannot be nested.");
               }
               else {
                  in_object_begin = true;
                  // TODO - record start of new object with given name
                  if (directive->type.empty()) {
                     log_illegal_identifier(directive, "ObjectBegin directive cannot specify an empty name.");
                  }
                  
                  // has an object by that name already been defined?
                  {
                     std::shared_ptr<poly::mesh_geometry> previous_instance = nullptr;
                     // TODO find and use an exception-less hash table
                     try {
                        previous_instance = name_mesh_map.at(directive->type);
                     }
                     catch (...) {}
                     if (previous_instance != nullptr) {
                        log_illegal_identifier(directive, "ObjectBegin directive cannot specify [" + directive->type +
                                                          "]; that name has already been previously defined.");
                     }
                  }
                  
                  // create geometry and insert into the map
                  current_geometry = std::make_shared<mesh_geometry>();
                  name_mesh_map[directive->type] = current_geometry;
               }
               break;
            }
            case DirectiveIdentifier::ObjectEnd: {
               if (in_object_begin) {
                  in_object_begin = false;
                  current_geometry = nullptr;
               }
               else {
                  log_illegal_directive(directive, "ObjectEnd directive without previous matching ObjectBegin directive");
               }
               break;
            }
            case DirectiveIdentifier::ObjectInstance: {
               if (in_object_begin) {
                  log_illegal_directive(directive, "ObjectInstance directive cannot appear between ObjectBegin and ObjectEnd directives");
               } 
               else if (current_geometry != nullptr) {
                  ERROR("BUG: current geometry isn't null, but should be");
               }
               std::string object_name = directive->type;
               if (object_name.empty()) {
                  log_illegal_identifier(directive, "ObjectInstance directive cannot specify an empty name.");
               }
               // get mesh geometry from hash table
               std::shared_ptr<poly::mesh_geometry> geometry = nullptr;
               // TODO find and use an exception-less hash table
               try {
                  geometry = name_mesh_map.at(object_name);
               }
               catch (...) {
                  log_illegal_identifier(directive, "ObjectInstance directive specifies name [" + object_name + "]; but that name hasn't been defined yet.");
               }
               
               // create mesh with current geometry and add to scene
               std::shared_ptr<poly::Transform> activeInverse = std::make_shared<poly::Transform>(activeTransform->Invert());
               poly::Mesh* mesh = new Mesh(activeTransform, activeInverse, activeMaterial, geometry);
               if (activeLight != nullptr) {
                  // TODO
                  mesh->spd = activeLight;
                  scene->Lights.push_back(mesh);
               }
               else {
                  mesh->material = activeMaterial;
               }
               scene->Shapes.push_back(mesh);
               break;
            }
            case DirectiveIdentifier::Rotate: {
               *activeTransform *= *rotate_directive(directive);
               break;
            }
            case DirectiveIdentifier::Scale: {
               *activeTransform *= *scale_directive(directive);
               break;
            }
            case DirectiveIdentifier::Shape: {
               ShapeIdentifier identifier;
               try {
                  identifier = ShapeIdentifierMap.at(directive->type);
               }
               catch (...) {
                  LogUnknownIdentifier(directive);
                  continue;
               }

               std::shared_ptr<poly::Transform> activeInverse = std::make_shared<poly::Transform>(activeTransform->Invert());
               
               switch (identifier) {
                  case ShapeIdentifier::objmesh: {
                     // make sure it has a filename argument
                     bool filenameMissing = true;
                     std::string mesh_filename;
                     for (const pbrt_argument &argument : directive->arguments) {
                        OBJMeshArgument arg;
                        try {
                           arg = OBJMeshArgumentMap.at(argument.Name);
                        }
                        catch (...) {
                           LogUnknownArgument(argument);
                           continue;
                        }
                        switch (arg) {
                           case Filename: {
                              filenameMissing = false;
                              mesh_filename = *argument.string_value;
                              if (argument.Type != pbrt_argument::pbrt_string) {
                                 LogWrongArgumentType(directive, argument);
                              }
                              break;
                           }
                           default:
                              LogUnknownArgument(argument);
                              continue;
                        }
                     }
                     if (filenameMissing) {
                        LogMissingArgument(directive, str::filename);
                        break;
                     }

                     const obj_parser parser;
                     std::shared_ptr<poly::mesh_geometry> geometry;
                     if (in_object_begin) {
                        geometry = current_geometry;
                     }
                     else {
                        geometry = std::make_shared<poly::mesh_geometry>();
                     }
                     scene->num_mesh_geometries++;
                     const std::string absolute_path = _basePathFromCWD + mesh_filename;
                     parser.parse_file(geometry, absolute_path);

                     if (in_object_begin) {
                        // maybe not necessary
                        current_geometry = geometry;
                     }
                     else {
                        poly::Mesh *mesh = new Mesh(activeTransform, activeInverse, activeMaterial, geometry);

                        if (activeLight != nullptr) {
                           // TODO
                           mesh->spd = activeLight;
                           scene->Lights.push_back(mesh);
                        } else {
                           mesh->material = activeMaterial;
                        }
                        scene->Shapes.push_back(mesh);
                     }
                     break;
                  
                  }
                  case ShapeIdentifier::plymesh: {
                     // make sure it has a filename argument
                     bool filenameMissing = true;
                     std::string mesh_filename;
                     for (const pbrt_argument& argument : directive->arguments) {
                        PLYMeshArgument arg;
                        try {
                           arg = PLYMeshArgumentMap.at(argument.Name);
                        }
                        catch (...) {
                           LogUnknownArgument(argument);
                           continue;
                        }
                        switch (arg) {
                           case PLYMeshArgument::Filename: {
                              filenameMissing = false;
                              mesh_filename = *argument.string_value;
                              if (argument.Type != pbrt_argument::pbrt_string) {
                                 LogWrongArgumentType(directive, argument);
                              }
                              break;
                           }
                           default:
                              LogUnknownArgument(argument);
                              continue;
                        }
                     }
                     if (filenameMissing) {
                        LogMissingArgument(directive, str::filename);
                        break;
                     }

                     const ply_parser parser;
                     std::shared_ptr<poly::mesh_geometry> geometry;
                     if (in_object_begin) {
                        geometry = current_geometry;
                     }
                     else {
                        geometry = std::make_shared<poly::mesh_geometry>();
                     }
                     scene->num_mesh_geometries++;
                     
                     const std::string absolute_path = _basePathFromCWD + mesh_filename;
                     parser.parse_file(geometry, absolute_path);

                     if (in_object_begin) {
                        // maybe not necessary
                        current_geometry = geometry;
                     }
                     else {
                        poly::Mesh* mesh = new Mesh(activeTransform, activeInverse, activeMaterial, geometry);
                        //mesh->Bound();
                        //mesh->CalculateVertexNormals();

                        if (activeLight != nullptr) {
                           // TODO
                           mesh->spd = activeLight;
                           scene->Lights.push_back(mesh);
                        }
                        else {
                           mesh->material = activeMaterial;
                        }
                        scene->Shapes.push_back(mesh);
                     }
                     break;
                  }
                  case ShapeIdentifier::sphere: {
                     for (const pbrt_argument& argument : directive->arguments) {
                        if (argument.Type == pbrt_argument::pbrt_argument_type::pbrt_float) {
                           const float radius = argument.float_values->at(0);
                           
                           poly::Transform radius_transform = Transform::Scale(radius);
                           std::shared_ptr<poly::Transform> temp_radius_transform = std::make_shared<poly::Transform>((*activeTransform) * radius_transform);
                           std::shared_ptr<poly::Transform> temp_radius_inverse = std::make_shared<poly::Transform>(temp_radius_transform->Invert());
                           
                           std::shared_ptr<poly::mesh_geometry> geometry = std::make_shared<poly::mesh_geometry>();
                           scene->num_mesh_geometries++;
                           const int subdivisions = std::max((int)radius, 20);
                           poly::SphereTesselator::Create(subdivisions, subdivisions, geometry);

                           if (in_object_begin) {
                              current_geometry = geometry;
                           }
                           else {
                              poly::Mesh* mesh = new poly::Mesh(temp_radius_transform, temp_radius_inverse, activeMaterial, geometry);
                              mesh->object_to_world = activeTransform;
                              mesh->world_to_object = activeInverse;

                              if (activeLight != nullptr) {
                                 mesh->spd = activeLight;
                                 scene->Lights.push_back(mesh);
                              }
                              else {
                                 mesh->material = activeMaterial;
                              }
                              scene->Shapes.push_back(mesh);
                           } 
                        }
                     }
                     break;
                  }
                  default: {
                     LogUnimplementedDirective(directive);
                     break;
                  }
               }
               break;
            }
            case DirectiveIdentifier::Transform: {
               *activeTransform *= *transform_directive(directive);
               break;
            }
            case DirectiveIdentifier::TransformBegin: {
               // push onto transform stack
               assert (activeTransform != nullptr);
               transformStack.push(activeTransform);
               activeTransform = std::make_shared<poly::Transform>(*(activeTransform.get()));
               break;
            }
            case DirectiveIdentifier::TransformEnd: {
               // pop transform stack
               if (!transformStack.empty()) {
                  std::shared_ptr<poly::Transform> stackValue = transformStack.top();
                  transformStack.pop();
                  assert (stackValue != nullptr);
                  activeTransform = stackValue;
               }
               break;
            }
            case DirectiveIdentifier::Translate: {
               *activeTransform *= *translate_directive(directive);
               break;
            }
            // TODO - other transform directives
            default: {
               LogUnimplementedDirective(directive);
               break;
            }
         }
      }

      integrator->Scene = scene;

      return std::make_unique<TileRunner>(
            std::move(sampler),
            scene,
            std::move(integrator),
            std::move(film),
            bounds,
            numSamples
      );
   }
   
}
