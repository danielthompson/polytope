//
// Created by Daniel on 07-Apr-18.
//

#include <sstream>
#include <stack>
#include <cassert>
#include <unordered_map>
#include "PBRTFileParser.h"
#include "mesh/OBJParser.h"
#include "mesh/PLYParser.h"
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
#include "../../cpu/shapes/linear_soa/mesh_linear_soa.h"
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
         const std::string NamedMaterial = "NamedMaterial";
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
         const std::string TransformBegin = "TransformBegin";
         const std::string TransformEnd = "TransformEnd";
         const std::string Translate = "Translate";
         const std::string WorldBegin = "WorldBegin";
         const std::string WorldEnd = "WorldEnd";
      }
      
      std::string _inputFilename = "";
      std::string _basePathFromCWD = "";

      // strings

      enum DirectiveName {
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
         PixelFilter,
         Rotate,
         Sampler,
         Scale,
         Shape,
         Translate,
         TransformBegin,
         TransformEnd,
         WorldBegin,
         WorldEnd
      };

      const std::unordered_map<std::string, DirectiveName> SceneDirectiveMap {
            {str::Camera, Camera},
            {str::Film, Film},
            {str::Integrator, Integrator},
            {str::LookAt, LookAt},
            {str::PixelFilter, PixelFilter},
            {str::Rotate, Rotate},
            {str::Sampler, Sampler},
            {str::Scale, Scale},
            {str::Translate, Translate}
      };

      const std::unordered_map<std::string, DirectiveName> WorldDirectiveMap {
            {str::AreaLightSource, AreaLightSource},
            {str::AttributeBegin, AttributeBegin},
            {str::AttributeEnd, AttributeEnd},
            {str::LightSource, LightSource},
            {str::MakeNamedMaterial, MakeNamedMaterial},
            {str::Material, Material},
            {str::NamedMaterial, NamedMaterial},
            {str::Rotate, Rotate},
            {str::Scale, Scale},
            {str::Shape, Shape},
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
      

      bool IsQuoted(const std::string &token) {
         return (token[0] == '"' && token[token.size() - 1] == '"');
      }

      bool StartQuoted(const std::string &token) {
         return (token[0] == '"' && token[token.size() - 1] != '"');
      }

      bool EndQuoted(std::string token) {
         return (token[0] != '"' && token[token.size() - 1] == '"');
      }

      void LogOther(const std::unique_ptr<PBRTDirective> &directive, const std::string &error) {
         Log.WithTime(directive->Name + ": " + error);
      }

      void LogMissingArgument(const std::unique_ptr<PBRTDirective>& directive, const std::string& argument) {
         Log.WithTime("Directive [" + directive->Name + "] w/ identifier [" + directive->Identifier + "] is missing argument [" + argument + "]");
      }

      void LogMissingDirective(const std::string& name, std::string& defaultOption) {
         Log.WithTime("Directive [" + name + "] is missing, defaulting to " + defaultOption + ".");
      }

      void LogUnknownDirective(const std::unique_ptr<PBRTDirective> &directive) {
         Log.WithTime("Directive [" + directive->Name + "] found, but is unknown. Ignoring.");
      }

      void LogUnknownIdentifier(const std::unique_ptr<PBRTDirective> &directive) {
         Log.WithTime("Directive [" + directive->Name + "] has unknown identifier [" + directive->Identifier + "].");
      }

      void LogUnknownArgument(const PBRTArgument &argument) {
         
         Log.WithTime("Unknown argument type/name combination: [" +
                            PBRTArgument::get_argument_type_string(argument.Type) + "] / [" + argument.Name + "].");
      }

      void LogWrongArgumentType(const std::unique_ptr<PBRTDirective> &directive, const PBRTArgument &argument) {
         Log.WithTime("Directive [" + directive->Name + "] w/ identifier [" + directive->Identifier + "] found has argument [" + argument.Name + "] with wrong type [" +
                            PBRTArgument::get_argument_type_string(argument.Type) + "].");
      }

      void LogUnimplementedDirective(const std::unique_ptr<PBRTDirective> &directive) {
         Log.WithTime("Directive [" + directive->Name + "] w/ identifier [" + directive->Identifier + "] found, but is not yet implemented. Ignoring.");
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

   std::unique_ptr<AbstractRunner> PBRTFileParser::ParseFile(const std::string &filepath) {

      std::unique_ptr<std::vector<std::vector<std::string>>> tokens = Scan(open_ascii_stream(filepath));

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

      return Parse(std::move(tokens));
   }

   std::unique_ptr<AbstractRunner> PBRTFileParser::ParseString(const std::string &text) {
      auto tokens = Scan(std::make_unique<std::istringstream>(text));
      return Parse(std::move(tokens));
   }

   std::unique_ptr<std::vector<std::vector<std::string>>> PBRTFileParser::Scan(const std::unique_ptr<std::istream> &stream) {

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
      
      Log.WithTime("Scan complete.");
      return tokens;
   }

   void LexFloatArrayArgument(const std::vector<std::string>& line, const int expected_num_elements, poly::PBRTArgument *argument) {
      if (line.size() != expected_num_elements + 1) {
         throw std::invalid_argument(line[0] + " requires exactly " + std::to_string(expected_num_elements) + " arguments, but found " + std::to_string(line.size()));
      }

      argument->Type = PBRTArgument::pbrt_float;
      argument->float_values = std::make_unique<std::vector<float>>();
      for (int i = 1; i <= expected_num_elements; i++) {
         std::string current_arg = line[i];
         float value;
         try {
            value = stof(current_arg);
         }
         catch (const std::invalid_argument &) {
            throw std::invalid_argument(line[0] + ": failed to parse [" + current_arg + "] as a float");
         }
         argument->float_values->push_back(value);
      }
   }
   
   std::unique_ptr<poly::PBRTDirective> PBRTFileParser::Lex(std::vector<std::string> line) {
      std::unique_ptr<poly::PBRTDirective> directive = std::make_unique<PBRTDirective>();

      if (line.empty()) {
         Log.WithTime("Lexed empty line. Hmm...");
         return directive;
      }

      directive->Name = line[0];

      bool debug = false;
      if (directive->Name == str::AttributeBegin) {
         debug = true;
      }
      
      if (line.size() == 1) {
         Log.WithTime("Lexed directive [" + directive->Name + "]");
         return directive;
      }

      // first, lex directives that use non-uniform argument syntax
      if (directive->Name == str::LookAt) {
         directive->Arguments = std::vector<PBRTArgument>();
         directive->Arguments.emplace_back(poly::PBRTArgument::PBRTArgumentType::pbrt_float);
         LexFloatArrayArgument(line, 9, &(directive->Arguments[0]));
         Log.WithTime("Lexed directive [" + directive->Name + "]");
         return directive;
      }
      
      if (directive->Name == str::Rotate) {
         directive->Arguments = std::vector<PBRTArgument>();
         directive->Arguments.emplace_back(poly::PBRTArgument::PBRTArgumentType::pbrt_float);
         LexFloatArrayArgument(line, 4, &(directive->Arguments[0]));
         Log.WithTime("Lexed directive [" + directive->Name + "]");
         return directive;
      }

      if (directive->Name == str::Scale) {
         directive->Arguments = std::vector<PBRTArgument>();
         directive->Arguments.emplace_back(poly::PBRTArgument::PBRTArgumentType::pbrt_float);
         LexFloatArrayArgument(line, 3, &(directive->Arguments[0]));
         Log.WithTime("Lexed directive [" + directive->Name + "]");
         return directive;
      }
      
      if (directive->Name == str::Translate) {
         directive->Arguments = std::vector<PBRTArgument>();
         directive->Arguments.emplace_back(poly::PBRTArgument::PBRTArgumentType::pbrt_float);
         LexFloatArrayArgument(line, 3, &(directive->Arguments[0]));
         Log.WithTime("Lexed directive [" + directive->Name + "]");
         return directive;
      }
      
      if (IsQuoted(line[1])) {
         directive->Identifier = line[1].substr(1, line[1].length() - 2);
      } 
      else {
         throw std::invalid_argument(line[0] + ": second token (identifier) isn't quoted, but should be");
      }

      if (line.size() == 2) {
         Log.WithTime("Lexed directive [" + directive->Name + "]");
         return directive;
      }
      
      bool inValue = false;
      bool in_arg = false;
      int i = 2;
      int current_arg_index = -1;
      PBRTArgument* current_arg = nullptr;
      while (i < line.size()) {
         if (StartQuoted(line[i]) && EndQuoted(line[i + 1])) {
            // we're in an argument
            if (directive->Arguments.empty())
               directive->Arguments = std::vector<PBRTArgument>();
            const PBRTArgument::PBRTArgumentType type = PBRTArgument::get_argument_type(line[i].substr(1, line[i].length() - 1));
            directive->Arguments.emplace_back(PBRTArgument(type));
            current_arg_index++;
            current_arg = &(directive->Arguments[current_arg_index]);
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
            if (IsQuoted(line[i])) {
               // probably just string?
               assert(current_arg->int_values == nullptr);
               assert(current_arg->float_values == nullptr);
               
               std::string value = line[i].substr(1, line[i].length() - 2);
               
               switch (current_arg->Type) {
                  case PBRTArgument::PBRTArgumentType::pbrt_string: {
                     assert(current_arg->bool_value == nullptr);
                     current_arg->string_value = std::make_unique<std::string>(value);
                     break;
                  }
                  case PBRTArgument::PBRTArgumentType::pbrt_bool: {
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
                     assert(false);
                  }
               }
            } 
            else {
               switch (current_arg->Type) {
                  case PBRTArgument::pbrt_rgb:
                  case PBRTArgument::pbrt_float: {
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
                  case PBRTArgument::pbrt_int: {
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
         throw std::invalid_argument("Lexer: current line has invalid token [" + line[i] + "]");
      }

//      if (inValue) {
//         directive->Arguments.push_back(argument);
//      }

      Log.WithTime("Lexed directive [" + directive->Name + "]");

      return directive;
   }
   
   std::unique_ptr<AbstractRunner> PBRTFileParser::Parse(const std::unique_ptr<std::vector<std::vector<std::string>>> tokens) noexcept(false){
      std::vector<std::unique_ptr<PBRTDirective>> scene_directives;
      std::vector<std::unique_ptr<PBRTDirective>> world_directives;

      {
         std::vector<std::unique_ptr<PBRTDirective>>* current_directives = &scene_directives;
         for (const std::vector<std::string>& line : *tokens) {
            std::unique_ptr<PBRTDirective> directive = Lex(line);
            // TODO ensure directive is valid for scene/world
            
            if (directive->Name == str::WorldBegin) {
               current_directives = &world_directives;
            }
            
            current_directives->push_back(std::move(directive));
         }
      }

      // sampler

      bool missingSampler = true;

      unsigned int numSamples = DefaultSamples;

      for (const std::unique_ptr<PBRTDirective> &directive : scene_directives) {
         if (directive->Name == str::Sampler) {
            missingSampler = false;
            if (directive->Identifier == str::halton) {
               Sampler = std::make_unique<GridSampler>();
            } else {
               LogUnknownIdentifier(directive);
               Sampler = std::make_unique<GridSampler>();
            }

            for (const PBRTArgument& arg : directive->Arguments) {
               if (arg.Type == PBRTArgument::pbrt_int) {
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

      if (Sampler == nullptr) {
         Sampler = std::make_unique<CenterSampler>();
      }

      Log.WithTime("Made (center) sampler.");

      // film

      bool missingFilm = true;

      for (const std::unique_ptr<PBRTDirective> &directive : scene_directives) {
         if (directive->Name == str::Film) {
            if (directive->Identifier == str::image) {
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

               for (const PBRTArgument& arg : directive->Arguments) {
                  if (arg.Type == PBRTArgument::pbrt_int) {
                     if (arg.Name == "xresolution") {
                        x = arg.int_values->at(0);
                        foundX = true;
                     } else if (arg.Name == "yresolution") {
                        y = arg.int_values->at(0);
                        foundY = true;
                     } else {
                        LogUnknownArgument(arg);
                     }
                  } else if (arg.Type == PBRTArgument::pbrt_string) {
                     if (arg.Name == "filename") {
                        filename = *(arg.string_value);
                        // TODO complain if filename arg is provided but empty
                     } else {
                        LogUnknownArgument(arg);
                     }
                  }
               }

               _bounds.x = foundX ? x : DefaultBoundsX;
               _bounds.y = foundY ? y : DefaultBoundsY;

               missingFilm = false;

               _film = std::make_unique<PNGFilm>(_bounds, filename, std::move(_filter));
            }
            break;
         }
      }

      if (missingFilm) {
         std::string defaultOption = "PNGFilm with 640x480 and output filename polytope.png";
         LogMissingDirective(str::Film, defaultOption);
      }

      if (_film == nullptr) {
         std::string filename = "polytope.png";
         _bounds.x = DefaultBoundsX;
         _bounds.y = DefaultBoundsY;
         _film = std::make_unique<PNGFilm>(_bounds, filename, std::move(_filter));
      }

      Log.WithTime("Made film.");

      // filter

      bool missingFilter = true;

      for (const std::unique_ptr<PBRTDirective>& directive : scene_directives) {
         if (directive->Name == str::PixelFilter) {
            missingFilter = false;
            if (directive->Identifier == "box") {
               unsigned int xWidth = 0;
               unsigned int yWidth = 0;

               for (const PBRTArgument& arg : directive->Arguments) {
                  if (arg.Type == PBRTArgument::pbrt_int) {
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

      if (_filter == nullptr) {
         _filter = std::make_unique<BoxFilter>(_bounds);
         _film->Filter = std::move(_filter);
      }

      Log.WithTime("Made filter.");

      // camera

      std::unique_ptr<AbstractCamera> camera;

      {
         Point eye;
         Point lookAt;
         Vector up;

         Transform currentTransform;
         
         for (const std::unique_ptr<PBRTDirective>& directive : scene_directives) {
            if (directive->Name == str::LookAt) {
               const float eyeX = directive->Arguments[0].float_values->at(0);
               const float eyeY = directive->Arguments[0].float_values->at(1);
               const float eyeZ = directive->Arguments[0].float_values->at(2);

               eye = Point(eyeX, eyeY, eyeZ);

               const float lookAtX = directive->Arguments[0].float_values->at(3);
               const float lookAtY = directive->Arguments[0].float_values->at(4);
               const float lookAtZ = directive->Arguments[0].float_values->at(5);

               lookAt = Point(lookAtX, lookAtY, lookAtZ);

               const float upX = directive->Arguments[0].float_values->at(6);
               const float upY = directive->Arguments[0].float_values->at(7);
               const float upZ = directive->Arguments[0].float_values->at(8);

               up = Vector(upX, upY, upZ);

               Transform t = Transform::LookAt(eye, lookAt, up, false);
               currentTransform *= t;

               Log.WithTime("Found LookAt.");
               break;
            }
         }

         CameraSettings settings = CameraSettings(_bounds, DefaultCameraFOV);

         bool foundCamera = false;

         for (const std::unique_ptr<PBRTDirective> &directive : scene_directives) {
            if (directive->Name == str::Camera) {
               if (directive->Identifier == "perspective") {
                  float fov = DefaultCameraFOV;

                  foundCamera = true;

                  for (const PBRTArgument& arg : directive->Arguments) {
                     if (arg.Type == PBRTArgument::pbrt_float) {
                        if (arg.Name == "fov") {
                           fov = arg.float_values->at(0);
                        } 
                        else {
                           LogUnknownArgument(arg);
                        }
                     }
                  }

                  settings.FieldOfView = fov;

                  camera = std::make_unique<PerspectiveCamera>(settings, currentTransform, true);
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
            camera = std::make_unique<PerspectiveCamera>(settings, currentTransform, true);
         }
      }

      // integrator

      bool missingIntegrator = true;

      for (const std::unique_ptr<PBRTDirective>& directive : scene_directives) {
         if (directive->Name == str::Integrator) {
            if (directive->Identifier == "path") {
               unsigned int maxDepth = 5;

               missingIntegrator = false;

               bool missingDepth = true;

               for (const PBRTArgument& arg : directive->Arguments) {
                  if (arg.Type == PBRTArgument::pbrt_int) {
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
               _integrator = std::make_unique<poly::PathTraceIntegrator>(maxDepth);
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

      if (_integrator == nullptr) {
         _integrator = std::make_unique<PathTraceIntegrator>(5);
      }

      // world

      std::vector<std::shared_ptr<poly::Material>> namedMaterials;

      std::stack<std::shared_ptr<poly::Material>> materialStack;
      std::stack<std::shared_ptr<SpectralPowerDistribution>> lightStack;
      std::stack<std::shared_ptr<Transform>> transformStack;

      std::shared_ptr<poly::Material> activeMaterial;
      std::shared_ptr<SpectralPowerDistribution> activeLight;
      std::shared_ptr<Transform> activeTransform = std::make_shared<Transform>();
      
      _scene = new Scene(std::move(camera));

      for (const std::unique_ptr<PBRTDirective>& directive : world_directives) {
         DirectiveName name;
         try {
            name = WorldDirectiveMap.at(directive->Name);
         }
         catch (...) {
            LogUnknownDirective(directive);
            continue;
         }

         switch (name) {
            case DirectiveName::AreaLightSource: {
               // lights with geometry
               if (directive->Identifier != "diffuse") {
                  LogUnknownIdentifier(directive);
                  break;
               }
               for (const PBRTArgument& argument : directive->Arguments) {
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
            case DirectiveName::AttributeBegin: {
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
               activeTransform = std::make_shared<Transform>(*(activeTransform.get()));
               break;
            }
            case DirectiveName::AttributeEnd: {
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
                  std::shared_ptr<Transform> stackValue = transformStack.top();
                  transformStack.pop();
                  assert (stackValue != nullptr);
                  activeTransform = stackValue;
               }
               break;
            }
            case DirectiveName::LightSource: {
               // lights without geometry
               if (directive->Identifier == "infinite") {
                  for (const PBRTArgument& argument : directive->Arguments) {
                     if (argument.Name == "L") {
                        const float r = argument.float_values->at(0);
                        const float g = argument.float_values->at(1);
                        const float b = argument.float_values->at(2);

                        const poly::SpectralPowerDistribution spd(r * 255, g * 255, b * 255);

                        _scene->Skybox = std::make_unique<ColorSkybox>(spd);
                        break;
                     }
                  }
               }
               else {
                  LogUnimplementedDirective(directive);
               }
               break;
            }
            case DirectiveName::MakeNamedMaterial: {
               const std::string materialName = directive->Identifier;
               std::shared_ptr<AbstractBRDF> brdf;
               poly::ReflectanceSpectrum reflectanceSpectrum;
               for (const PBRTArgument &argument : directive->Arguments) {
                  if (argument.Type == PBRTArgument::pbrt_string) {
                     if (argument.Name == "type" && *(argument.string_value) == str::matte) {
                        // TODO instead of using default, parse values
                        ReflectanceSpectrum refl(0.5f, 0.5f, 0.5f);
                        brdf = std::make_unique<poly::LambertBRDF>(refl);
                     }
                  }
                  if (argument.Type == PBRTArgument::pbrt_rgb) {
                     if (argument.Name == str::Kd) {
                        reflectanceSpectrum.r = argument.float_values->at(0);
                        reflectanceSpectrum.g = argument.float_values->at(1);
                        reflectanceSpectrum.b = argument.float_values->at(2);
                     }
                  }
               }

               namedMaterials.push_back(std::make_shared<poly::Material>(std::move(brdf), reflectanceSpectrum, materialName));
               break;
            }
            case DirectiveName::Material: {
               MaterialIdentifier identifier;
               try {
                  identifier = MaterialIdentifierMap.at(directive->Identifier);
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

                     for (const PBRTArgument &argument : directive->Arguments) {
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
                              if (argument.Type == PBRTArgument::pbrt_rgb) {
                                 kd.r = argument.float_values->at(0);
                                 kd.g = argument.float_values->at(1);
                                 kd.b = argument.float_values->at(2);
                              }
                              break;
                           }
                           case Ks: {
                              if (argument.Type == PBRTArgument::pbrt_rgb) {
                                 ks.r = argument.float_values->at(0);
                                 ks.g = argument.float_values->at(1);
                                 ks.b = argument.float_values->at(2);
                              }
                              break;
                           }
                           case Roughness: {
                              if (argument.Type == PBRTArgument::pbrt_float) {
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
                     for (const PBRTArgument& argument : directive->Arguments) {
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
                        if (argument.Name == str::Kd && argument.Type == PBRTArgument::pbrt_rgb) {
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
                     for (const PBRTArgument& argument : directive->Arguments) {
                        if (argument.Name == str::Kr && argument.Type == PBRTArgument::pbrt_rgb) {
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
            case DirectiveName::NamedMaterial: {
               std::string materialName = directive->Identifier;
               bool found = false;
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
            case DirectiveName::Rotate: {
               // TODO need to ensure just 1 argument with 4 values
               PBRTArgument* arg = &(directive->Arguments[0]);
               const float angle = arg->float_values->at(0) * PIOver180;
               float x = arg->float_values->at(1);
               float y = arg->float_values->at(2);
               float z = arg->float_values->at(3);

               // normalize
               const float oneOverLength = 1.f / std::sqrt(x * x + y * y + z * z);
               x *= oneOverLength;
               y *= oneOverLength;
               z *= oneOverLength;

               Transform t = Transform::Rotate(angle, x, y, z);

               assert (activeTransform != nullptr);
               Transform *active = activeTransform.get();
               *active *= t;
               break;
            }
            case DirectiveName::Scale: {
               PBRTArgument* arg = &(directive->Arguments[0]);
               float x = arg->float_values->at(0);
               float y = arg->float_values->at(1);
               float z = arg->float_values->at(2);

               Transform t = Transform::Scale(x, y, z);

               assert (activeTransform != nullptr);
               Transform *active = activeTransform.get();
               *active *= t;
               break;
            }
            case DirectiveName::Shape: {
               ShapeIdentifier identifier;
               try {
                  identifier = ShapeIdentifierMap.at(directive->Identifier);
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
                     for (const PBRTArgument& argument : directive->Arguments) {
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
                              if (argument.Type != PBRTArgument::pbrt_string) {
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

                     TMesh* mesh = new TMesh(activeTransform, activeInverse, activeMaterial);

                     const OBJParser<TMesh> parser;
                     const std::string absoluteObjFilepath = _basePathFromCWD + mesh_filename;
                     parser.ParseFile(mesh, absoluteObjFilepath);
                     mesh->Bound();
                     mesh->CalculateVertexNormals();
                     //mesh->ObjectToWorld = *activeTransform;

                     if (activeLight != nullptr) {
                        // TODO
                        mesh->spd = activeLight;
                        _scene->Lights.push_back(mesh);
                     }
                     else {
                        mesh->Material = activeMaterial;
                     }
                     _scene->Shapes.push_back(mesh);
                     break;
                  }
                  case ShapeIdentifier::plymesh: {
                     // make sure it has a filename argument
                     bool filenameMissing = true;
                     std::string mesh_filename;
                     for (const PBRTArgument& argument : directive->Arguments) {
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
                              if (argument.Type != PBRTArgument::pbrt_string) {
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

                     TMesh* mesh = new TMesh(activeTransform, activeInverse, activeMaterial);

                     const PLYParser<TMesh> parser;
                     const std::string absoluteObjFilepath = /*GetCurrentWorkingDirectory() + UnixPathSeparator +*/ _basePathFromCWD + mesh_filename;
                     parser.ParseFile(mesh, absoluteObjFilepath);
                     //mesh->Bound();
                     mesh->CalculateVertexNormals();

                     if (activeLight != nullptr) {
                        // TODO
                        mesh->spd = activeLight;
                        _scene->Lights.push_back(mesh);
                     }
                     else {
                        mesh->Material = activeMaterial;   
                     }
                     _scene->Shapes.push_back(mesh);
                     break;
                  }
                  case ShapeIdentifier::sphere: {
                     for (const PBRTArgument& argument : directive->Arguments) {
                        if (argument.Type == PBRTArgument::PBRTArgumentType::pbrt_float) {
                           const float radius = argument.float_values->at(0);
                           
                           poly::Transform radius_transform = Transform::Scale(radius);
                           std::shared_ptr<poly::Transform> temp_radius_transform = std::make_shared<poly::Transform>((*activeTransform) * radius_transform);
                           std::shared_ptr<poly::Transform> temp_radius_inverse = std::make_shared<poly::Transform>(temp_radius_transform->Invert());
                           
                           poly::TMesh* mesh = new MeshLinearSOA(temp_radius_transform, temp_radius_inverse, activeMaterial);
                           const int subdivisions = std::max((int)radius, 20);
                           poly::SphereTesselator::Create(subdivisions, subdivisions, mesh);
                           mesh->ObjectToWorld = activeTransform;
                           mesh->WorldToObject = activeInverse;

                           if (activeLight != nullptr) {
                              mesh->spd = activeLight;
                              _scene->Lights.push_back(mesh);
                           }
                           else {
                              mesh->Material = activeMaterial;
                           }
                           _scene->Shapes.push_back(mesh);
                        }
                     }
                  }
                  default: {
                     LogUnimplementedDirective(directive);
                     break;
                  }
               }
               break;
            }
            case DirectiveName::TransformBegin: {
               // push onto transform stack
               assert (activeTransform != nullptr);
               transformStack.push(activeTransform);
               activeTransform = std::make_shared<Transform>(*(activeTransform.get()));
               break;
            }
            case (DirectiveName::TransformEnd): {
               // pop transform stack
               if (!transformStack.empty()) {
                  std::shared_ptr<Transform> stackValue = transformStack.top();
                  transformStack.pop();
                  assert (stackValue != nullptr);
                  activeTransform = stackValue;
               }
               break;
            }
            case DirectiveName::Translate: {
               // need to ensure just one argument with 3 values
               PBRTArgument* arg = &(directive->Arguments[0]);
               float x = arg->float_values->at(0);
               float y = arg->float_values->at(1);
               float z = arg->float_values->at(2);
               Transform t = Transform::Translate(x, y, z);

               assert (activeTransform != nullptr);
               Transform *active = activeTransform.get();
               *active *= t;
               break;
            }
            // TODO - other transform directives
            default: {
               LogUnimplementedDirective(directive);
               break;
            }
         }
      }

      _integrator->Scene = _scene;

      return std::make_unique<TileRunner>(
         std::move(Sampler),
         _scene,
         std::move(_integrator),
         std::move(_film),
         _bounds,
         numSamples
      );
   }
   
}
