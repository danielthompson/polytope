//
// Created by Daniel on 07-Apr-18.
//

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stack>
#include <map>
#include "PBRTFileParser.h"
#include "../integrators/PathTraceIntegrator.h"
#include "../samplers/HaltonSampler.h"
#include "../runners/TileRunner.h"
#include "../films/PNGFilm.h"
#include "../filters/BoxFilter.h"
#include "../cameras/PerspectiveCamera.h"
#include "../shading/brdf/LambertBRDF.h"
#include "../shapes/Sphere.h"
#include "../scenes/NaiveScene.h"
#include "../utilities/Common.h"

namespace Polytope {

   namespace {
      std::string _inputFilename = "";

      // datatypes

      const std::string IntegerText = "integer";
      const std::string FloatText = "float";
      const std::string StringText = "string";
      const std::string RGBText = "rgb";

      // scene

      const std::string CameraText = "Camera";
      const std::string FilmText = "Film";
      const std::string IntegratorText = "Integrator";
      const std::string LookAtText = "LookAt";
      const std::string PixelFilterText = "PixelFilter";
      const std::string SamplerText = "Sampler";

      // world

      const std::string AreaLightSourceText = "AreaLightSource";
      const std::string AttributeBeginText = "AttributeBegin";
      const std::string AttributeEndText = "AttributeEnd";
      const std::string MakeNamedMaterialText = "MakeNamedMaterial";
      const std::string NamedMaterialText = "NamedMaterial";
      const std::string ShapeText = "Shape";
      const std::string TransformBeginText = "TransformBegin";
      const std::string TransformEndText = "TransformEnd";
      const std::string TranslateText = "Translate";
      const std::string WorldBeginText = "WorldBegin";
      const std::string WorldEndText = "WorldEnd";

      const std::vector<std::string> Directives {
         AreaLightSourceText,
         AttributeBeginText, // done
         AttributeEndText, // done
         CameraText,
         FilmText,
         IntegratorText,
         LookAtText, // done
         MakeNamedMaterialText,
         NamedMaterialText, // done
         PixelFilterText,
         SamplerText,
         ShapeText,
         TransformBeginText, // done
         TransformEndText, // done
         TranslateText, // done
         WorldBeginText, // done
         WorldEndText // done
      };

      enum SceneDirectiveName {
         Camera,
         Film,
         Integrator,
         LookAt,
         PixelFilter,
         Sampler
      };

      std::map<std::string, SceneDirectiveName> SceneDirectiveMap {
         {"Camera", Camera},
         {"Film", Film},
         {"Integrator", Integrator},
         {"LookAt", LookAt},
         {"PixelFilter", PixelFilter},
         {"Sampler", Sampler}
      };

      void LogBadArgument(const PBRTArgument &argument) {
         Log.Log("Unknown argument type/name combination: [" + argument.Type + "] / [" + argument.Name + "].");
      }

      void LogMissingArgument(const PBRTDirective &directive, const std::string& argument) {
         Log.Log("Directive [" + directive.Name + "] w/ identifier [" + directive.Identifier + "] is missing argument [" + argument + "]");
      }

      unsigned int stoui(const std::string& text) {
         return static_cast<unsigned int>(stoi(text));
      }
   }

   std::unique_ptr<AbstractRunner> PBRTFileParser::ParseFile(const std::string &filepath) {

      std::string cwd = GetCurrentWorkingDirectory();
      std::string absolutePath = cwd + "//" + filepath;

      Log.WithTime("Trying to open [" + absolutePath + "]...");
      std::vector<std::vector<std::string>> tokens;
      {
         Log.WithTime("Creating stream on [" + absolutePath + "]...");
         std::unique_ptr<std::istream> stream = std::make_unique<std::ifstream>(filepath);

         if (!stream->good()) {
            throw std::invalid_argument("Couldn't open stream on [" + absolutePath + "].");
         }

         tokens = Scan(std::move(stream));
      }

      // determine the name of the file from the given path

      const auto lastPos = filepath.find_last_of('/');
      if (lastPos == std::string::npos) {
         _inputFilename = filepath;
      }
      else {
         _inputFilename = filepath.substr(lastPos + 1);
      }

      return Parse(tokens);
   }

   std::unique_ptr<AbstractRunner> PBRTFileParser::ParseString(const std::string &text) {
      auto tokens = Scan(std::make_unique<std::istringstream>(text));
      return Parse(tokens);
   }

   std::vector<std::vector<std::string>> PBRTFileParser::Scan(std::unique_ptr<std::istream> stream) {

      std::vector<std::vector<std::string>> tokens;

      int sourceLineNumber = 0;
      int targetLineNumber = -1;
      std::string line;
      while (getline(*stream, line)) {
         tokens.emplace_back();
         std::string word;
         std::istringstream iss(line, std::istringstream::in);

         while (iss >> word) {
            // strip out comments
            if (word.find('#') == 0)
               break;

            if (std::find(Directives.begin(), Directives.end(), word) != Directives.end()) {
               // if this is a directive, then we move on to a new line
               targetLineNumber++;
            }

            // split brackets, if needed

            if (word.size() > 1) {
               const unsigned long lastIndex = word.size() - 1;

               if (word[0] == '[') {
                  tokens[targetLineNumber].push_back("[");
                  tokens[targetLineNumber].push_back(word.substr(1, lastIndex));
               } else if (word[lastIndex] == ']') {
                  tokens[targetLineNumber].push_back(word.substr(0, lastIndex - 1));
                  tokens[targetLineNumber].push_back("]");
               } else {
                  tokens[targetLineNumber].push_back(word);
               }
            } else {
               tokens[targetLineNumber].push_back(word);
            }

         }

         sourceLineNumber++;

      }

      return tokens;
   }

   std::unique_ptr<AbstractRunner> PBRTFileParser::Parse(std::vector<std::vector<std::string>> tokens) noexcept(false){
      std::vector<PBRTDirective> sceneDirectives;
      std::vector<PBRTDirective> worldDirectives;

      {
         std::vector<PBRTDirective> *currentDirectives = &sceneDirectives;

         for (std::vector<std::string> line : tokens) {
            PBRTDirective currentDirective = PBRTDirective();

            if (line.empty())
               continue;

            currentDirective.Name = line[0];

            if (currentDirective.Name == WorldBeginText)
               currentDirectives = &worldDirectives;

            if (line.size() == 1) {
               //currentDirective.
               currentDirectives->push_back(currentDirective);

               continue;
            }

            if (IsQuoted(line[1])) {
               currentDirective.Identifier = line[1].substr(1, line[1].length() - 2);
            } else {
               currentDirective.Arguments = std::vector<PBRTArgument>();
               PBRTArgument argument = PBRTArgument();
               argument.Type = "float";
               argument.Values = std::vector<std::string>();

               for (int i = 1; i < line.size(); i++) {
                  argument.Values.push_back(line[i]);
               }

               currentDirective.Arguments.push_back(argument);
               currentDirectives->push_back(currentDirective);
               continue;
            }

            if (line.size() == 2) {
               currentDirectives->push_back(currentDirective);
               continue;
            }

            currentDirective.Arguments = std::vector<PBRTArgument>();
            PBRTArgument currentArgument = PBRTArgument();
            bool inValue = false;
            int i = 2;
            while (i < line.size()) {

               if (StartQuoted(line[i]) && EndQuoted(line[i + 1])) {
                  // we're in an argument
                  currentArgument.Type = line[i].substr(1, line[i].length() - 1);
                  currentArgument.Name = line[i + 1].substr(0, line[i + 1].length() - 1);
                  inValue = true;
                  i += 2;
                  continue;
               }
               if (line[i] == "[") {
                  inValue = true;
                  i++;
                  continue;
               }
               if (line[i] == "]") {
                  inValue = false;
                  i++;
                  currentDirective.Arguments.push_back(currentArgument);
                  currentArgument = PBRTArgument();
                  continue;
               }
               if (inValue) {
                  if (IsQuoted(line[i])) {
                     currentArgument.Values.push_back(line[i].substr(1, line[i].length() - 2));
                  } else {
                     currentArgument.Values.push_back(line[i]);
                  }
                  i++;
                  continue;
               }
            }

            if (inValue) {
               currentDirective.Arguments.push_back(currentArgument);
            }

            currentDirectives->push_back(currentDirective);
         }
      }

      // sampler

      unsigned int numSamples = 16;

      for (const PBRTDirective& directive : sceneDirectives) {
         if (directive.Name == SamplerText) {
            if (directive.Identifier == "halton") {
               _sampler = std::make_unique<HaltonSampler>();
               break;
            } else {
               LogBadIdentifier(directive);
               _sampler = std::make_unique<HaltonSampler>();
            }

            for (const PBRTArgument& arg : directive.Arguments) {
               if (arg.Type == IntegerText) {
                  if (arg.Name == "pixelsamples") {
                     numSamples = stoui(arg.Values[0]);
                     break;
                  } else {
                     LogBadArgument(arg);
                  }
                  break;
               }
            }
            break;
         }
      }

      if (_sampler == nullptr) {
         Log.Log("No Sampler specified, using Halton.");
         _sampler = std::make_unique<HaltonSampler>();
      }

      // filter

      bool createBoxFilter = false;

      for (const PBRTDirective& directive : sceneDirectives) {
         if (directive.Name == PixelFilterText) {
            if (directive.Identifier == "box") {
               createBoxFilter = true;
               unsigned int xWidth = 0;
               unsigned int yWidth = 0;

               for (const PBRTArgument& arg : directive.Arguments) {
                  if (arg.Type == IntegerText) {
                     if (arg.Name == "xwidth") {
                        xWidth = stoui(arg.Values[0]);
                     } else if (arg.Name == "ywidth") {
                        yWidth = stoui(arg.Values[0]);
                     } else {
                        LogBadArgument(arg);
                     }
                     break;
                  }
               }
            } else {
               LogBadIdentifier(directive);
            }
            break;
         }
      }

      // film

      Polytope::Bounds bounds;

      for (const PBRTDirective& directive : sceneDirectives) {
         if (directive.Name == FilmText) {
            if (directive.Identifier == "image") {
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

               for (const PBRTArgument& arg : directive.Arguments) {
                  if (arg.Type == IntegerText) {
                     if (arg.Name == "xresolution") {
                        x = stoui(arg.Values[0]);
                        foundX = true;
                     } else if (arg.Name == "yresolution") {
                        y = stoui(arg.Values[0]);
                        foundY = true;
                     } else {
                        LogBadArgument(arg);
                     }
                  } else if (arg.Type == StringText) {
                     if (arg.Name == "filename") {
                        filename = arg.Values[0];
                     } else {
                        LogBadArgument(arg);
                     }
                  }
               }

               bounds.x = foundX ? x : 640;
               bounds.y = foundY ? y : 480;

               if (createBoxFilter) {
                  _filter = std::make_unique<BoxFilter>(bounds);
               }
               _film = std::make_unique<PNGFilm>(bounds, filename, std::move(_filter));
            }
            break;
         }
      }

      // camera

      std::unique_ptr<AbstractCamera> camera;

      {
         Transform currentTransform;

         for (const PBRTDirective& directive : sceneDirectives) {
            if (directive.Name == LookAtText) {
               if (directive.Arguments.size() == 1) {
                  if (directive.Arguments[0].Values.size() == 9) {
                     float eyeX = stof(directive.Arguments[0].Values[0]);
                     float eyeY = stof(directive.Arguments[0].Values[1]);
                     float eyeZ = stof(directive.Arguments[0].Values[2]);

                     Point eye = Point(eyeX, eyeY, eyeZ);

                     float lookAtX = stof(directive.Arguments[0].Values[3]);
                     float lookAtY = stof(directive.Arguments[0].Values[4]);
                     float lookAtZ = stof(directive.Arguments[0].Values[5]);

                     Point lookAt = Point(lookAtX, lookAtY, lookAtZ);

                     float upX = stof(directive.Arguments[0].Values[6]);
                     float upY = stof(directive.Arguments[0].Values[7]);
                     float upZ = stof(directive.Arguments[0].Values[8]);

                     Vector up = Vector(upX, upY, upZ);

                     Transform t = Transform::LookAt(eye, lookAt, up);

                     currentTransform = currentTransform * t;
                  }
               }
               break;
            }
         }

         CameraSettings settings = CameraSettings(bounds, 50);

         for (const PBRTDirective &directive : sceneDirectives) {
            if (directive.Name == CameraText) {
               if (directive.Identifier == "perspective") {
                  float fov = 50;

                  for (const PBRTArgument& arg : directive.Arguments) {
                     if (arg.Type == FloatText) {
                        if (arg.Name == "fov") {
                           // TODO remove static_cast?
                           fov = static_cast<float>(stof(arg.Values[0]));
                        } else {
                           LogBadArgument(arg);
                        }
                     }
                  }

                  settings.FieldOfView = fov;

                  camera = std::make_unique<PerspectiveCamera>(settings, currentTransform);
               }
               break;
            }
         }
      }

      // integrator

      for (const PBRTDirective& directive : sceneDirectives) {
         if (directive.Name == IntegratorText) {
            if (directive.Identifier == "path") {
               unsigned int maxDepth = 5;

               bool missingDepth = true;

               for (const PBRTArgument& arg : directive.Arguments) {
                  if (arg.Type == IntegerText) {
                     if (arg.Name == "maxdepth") {
                        maxDepth = stoui(arg.Values[0]);
                        missingDepth = false;
                        break;
                     } else {
                        LogBadArgument(arg);
                     }
                  }
               }

               if (missingDepth) {
                  LogMissingArgument(directive, "maxdepth");
               }
               _integrator = std::make_unique<PathTraceIntegrator>(maxDepth);
            }
            break;
         }
      }

      // world

      std::vector<std::shared_ptr<Material>> namedMaterials;

      std::stack<std::shared_ptr<Material>> materialStack;
      std::stack<std::shared_ptr<SpectralPowerDistribution>> lightStack;
      std::stack<std::shared_ptr<Transform>> transformStack;

      std::shared_ptr<Material> activeMaterial;
      std::shared_ptr<SpectralPowerDistribution> activeLight;
      std::shared_ptr<Transform> activeTransform;

      const std::shared_ptr<Material> materialMarker = std::make_shared<Material>(AttributeBeginText);
      const std::shared_ptr<SpectralPowerDistribution> lightMarker = std::make_shared<SpectralPowerDistribution>();
      const std::shared_ptr<Transform> transformMarker = std::make_shared<Transform>();

      _scene = new NaiveScene(std::move(camera));

      for (const PBRTDirective& directive : worldDirectives) {
         //Log.Log(std::string("Checking directive [") + directive.Name + "]");
         if (directive.Name == MakeNamedMaterialText) {
            std::string materialName = directive.Identifier;
            std::shared_ptr<AbstractBRDF> brdf;
            Polytope::ReflectanceSpectrum reflectanceSpectrum;
            for (const PBRTArgument &argument : directive.Arguments) {
               if (argument.Type == StringText) {
                  if (argument.Name == "type") {
                     if (argument.Values[0] == "matte") {
                        brdf = std::make_unique<Polytope::LambertBRDF>();
                     }
                  }
               }
               if (argument.Type == RGBText) {
                  if (argument.Name == "Kd") {
                     reflectanceSpectrum.r = stof(argument.Values[0]);
                     reflectanceSpectrum.g = stof(argument.Values[1]);
                     reflectanceSpectrum.b = stof(argument.Values[2]);
                  }
               }
            }

            namedMaterials.push_back(std::make_shared<Material>(std::move(brdf), reflectanceSpectrum, materialName));

         }

         else if (directive.Name == AttributeBeginText) {

            // push onto material stack
            materialStack.push(materialMarker);
            if (activeMaterial != nullptr) {
               materialStack.push(activeMaterial);
            }

            // push onto light stack
            lightStack.push(lightMarker);
            if (activeLight != nullptr) {
               lightStack.push(activeLight);
            }
            
            // push onto transform stack
            transformStack.push(transformMarker);
            if (activeTransform != nullptr) {
               transformStack.push(activeTransform);
            }
         }

         else if (directive.Name == AttributeEndText) {

            // pop material stack
            if (!materialStack.empty()) {
               std::shared_ptr<Material> stackValue = materialStack.top();
               materialStack.pop();
               
               if (stackValue == materialMarker) {
                  // no value was pushed, so there wasn't any active material before, so there shouldn't be now
                  activeMaterial = nullptr;
               }
               else {
                  // a value was pushed, so there should be at least one more materialMarker on the stack
                  if (materialStack.empty()) {
                     // OOPS, should never happen
                  }
                  else {
                     // restore the previously active material
                     activeMaterial = stackValue;
                     // pop the marker
                     materialStack.pop();
                  }
               }
            }

            // pop light stack
            if (!lightStack.empty()) {
               std::shared_ptr<SpectralPowerDistribution> stackValue = lightStack.top();
               lightStack.pop();

               if (stackValue == lightMarker) {
                  // no value was pushed, so there wasn't any active light before, so there shouldn't be now
                  activeLight = nullptr;
               }
               else {
                  // a value was pushed, so there should be at least one more lightMarker on the stack
                  if (lightStack.empty()) {
                     // OOPS, should never happen
                  }
                  else {
                     // restore the previously active light
                     activeLight = stackValue;
                     // pop the marker
                     lightStack.pop();
                  }
               }
            }

            // pop transform stack
            if (!transformStack.empty()) {
               std::shared_ptr<Transform> stackValue = transformStack.top();
               transformStack.pop();

               if (stackValue == transformMarker) {
                  // no value was pushed, so there wasn't any active transform before, so there shouldn't be now
                  activeTransform = nullptr;
               }
               else {
                  // a value was pushed, so there should be at least one more transformMarker on the stack
                  if (transformStack.empty()) {
                     // OOPS, should never happen
                  }
                  else {
                     // restore the previously active transform
                     activeTransform = stackValue;
                     // pop the marker
                     transformStack.pop();
                  }
               }
            }
         }

         else if (directive.Name == AreaLightSourceText) {
            for (const PBRTArgument& argument : directive.Arguments) {
               if (argument.Name == "L") {
                  if (activeLight == nullptr) {
                     activeLight = std::make_shared<SpectralPowerDistribution>();
                  }
                  activeLight->r = stof(argument.Values[0]);
                  activeLight->g = stof(argument.Values[1]);
                  activeLight->b = stof(argument.Values[2]);
                  break;
               }
            }
         }

         else if (directive.Name == TransformBeginText) {

            // push onto transform stack
            transformStack.push(transformMarker);
            if (activeTransform != nullptr) {
               transformStack.push(activeTransform);
            }
         }

         else if (directive.Name == TransformEndText) {
            // pop transform stack
            if (!transformStack.empty()) {
               std::shared_ptr<Transform> stackValue = transformStack.top();
               transformStack.pop();

               if (stackValue == transformMarker) {
                  // no value was pushed, so there wasn't any active transform before, so there shouldn't be now
                  activeTransform = nullptr;
               }
               else {
                  // a value was pushed, so there should be at least one more transformMarker on the stack
                  if (transformStack.empty()) {
                     // OOPS, should never happen
                  }
                  else {
                     // restore the previously active transform
                     activeTransform = stackValue;
                     // pop the marker
                     transformStack.pop();
                  }
               }
            }
         }

         else if (directive.Name == TranslateText) {
            // need to ensure just one argument with 3 values
            PBRTArgument argument = directive.Arguments[0];
            float x = std::stof(argument.Values[0]);
            float y = std::stof(argument.Values[1]);
            float z = std::stof(argument.Values[2]);
            Transform t = Transform::Translate(x, y, z);

            if (activeTransform == nullptr) {
               activeTransform = std::make_shared<Transform>();
            }

            Transform *active = activeTransform.get();
            *active *= t;
         }

         // other transform directives

         else if (directive.Name == ShapeText) {
            if (directive.Identifier == "sphere") {
               PBRTArgument argument = directive.Arguments[0];
               if (argument.Type == FloatText) {
                  float radius = std::stof(argument.Values[0]);
                  AbstractShape *sphere = new Sphere(*activeTransform, activeMaterial);
                  if (activeLight != nullptr) {
                     ShapeLight *sphereLight = new ShapeLight(*activeLight);
                     sphere->Light = sphereLight;
                     _scene->Lights.push_back(sphereLight);
                  }
                  _scene->Shapes.push_back(sphere);
               }
            }
            else {
               LogBadIdentifier(directive);
            }
         }

         else if (directive.Name == NamedMaterialText) {
            std::string materialName = directive.Identifier;
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
         }
      }

      _integrator->Scene = _scene;

      return std::make_unique<TileRunner>(
         std::move(_sampler),
         _scene,
         std::move(_integrator),
         std::move(_film),
         _bounds,
         numSamples
      );
   }




   void PBRTFileParser::LogBadIdentifier(const PBRTDirective &directive) {
      Log.Log("Unknown directive/identifier combination: [" + directive.Name + "] / [" + directive.Identifier + "].");
   }

   void PBRTFileParser::LogOther(const PBRTDirective &directive, const std::string &error) {
      Log.Log(directive.Name + ": " + error);
   }

   bool PBRTFileParser::IsQuoted(std::string token) {
      return (token[0] == '"' && token[token.size() - 1] == '"');
   }

   bool PBRTFileParser::StartQuoted(std::string token) {
      return (token[0] == '"' && token[token.size() - 1] != '"');
   }

   bool PBRTFileParser::EndQuoted(std::string token) {
      return (token[0] != '"' && token[token.size() - 1] == '"');
   }

   void PBRTFileParser::CreateSampler(std::vector<std::string> &directive)  {
      if (directive[1] == "\"halton\"") {
         _sampler = std::make_unique<HaltonSampler>();
      }
      else {
         throw std::invalid_argument("Given sampler [" + directive[1] + "] not supported");
      }

      _numSamples = std::stoi(directive[4]);
   }

   void PBRTFileParser::CreateIntegrator(std::vector<std::string> &directive) {
      if (directive[1] == "\"path\"") {
         _sampler = std::make_unique<HaltonSampler>();
      }
      else {
         throw std::invalid_argument("Given sampler [" + directive[1] + "] not supported");
      }

      _numSamples = std::stoi(directive[4]);
   }
}
