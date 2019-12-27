//
// Created by Daniel on 07-Apr-18.
//

#include <sstream>
#include <stack>
#include <map>
#include <algorithm>
#include "PBRTFileParser.h"
#include "mesh/OBJFileParser.h"
#include "../integrators/PathTraceIntegrator.h"
#include "../samplers/HaltonSampler.h"
#include "../runners/TileRunner.h"
#include "../films/PNGFilm.h"
#include "../filters/BoxFilter.h"
#include "../cameras/PerspectiveCamera.h"
#include "../shading/brdf/LambertBRDF.h"
#include "../scenes/NaiveScene.h"
#include "../utilities/Common.h"
#include "../scenes/skyboxes/ColorSkybox.h"
#include "../structures/Vectors.h"
#include "../samplers/CenterSampler.h"

namespace Polytope {

   namespace {
      std::string _inputFilename = "";
      std::string _basePathFromCWD = "";

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

      enum SceneDirectiveName {
         Camera,
         Film,
         Integrator,
         LookAt,
         PixelFilter,
         Sampler
      };

      const std::map<std::string, SceneDirectiveName> SceneDirectiveMap {
            {CameraText, Camera},
            {FilmText, Film},
            {IntegratorText, Integrator},
            {LookAtText, LookAt},
            {PixelFilterText, PixelFilter},
            {SamplerText, Sampler}
      };

      // world

      enum WorldDirectiveName {
         AreaLightSource,
         AttributeBegin,
         AttributeEnd,
         LightSource,
         MakeNamedMaterial,
         Material,
         NamedMaterial,
         Rotate,
         Shape,
         TransformBegin,
         TransformEnd,
         Translate,
         WorldBegin,
         WorldEnd
      };

      const std::map<std::string, WorldDirectiveName> WorldDirectiveMap {
            {"AreaLightSource", AreaLightSource},
            {"AttributeBegin", AttributeBegin},
            {"AttributeEnd", AttributeEnd},
            {"LightSource", LightSource},
            {"MakeNamedMaterial", MakeNamedMaterial},
            {"Material", Material},
            {"NamedMaterial", NamedMaterial},
            {"Rotate", Rotate},
            {"Shape", Shape},
            {"TransformBegin", TransformBegin},
            {"TransformEnd", TransformEnd},
            {"Translate", Translate},
            {"WorldBegin", WorldBegin},
            {"WorldEnd", WorldEnd}
      };

      enum ShapeIdentifier {
         Sphere,
         OBJMesh
      };

      const std::map<std::string, ShapeIdentifier> ShapeIdentifierMap {
            {"sphere", Sphere},
            {"objmesh", OBJMesh}
      };

      enum MaterialIdentifier {
         Matte
      };

      const std::map<std::string, MaterialIdentifier> MaterialIdentifierMap {
            {"matte", Matte}
      };

      enum OBJMeshArgument {
         Filename
      };

      const std::map<std::string, OBJMeshArgument> OBJMeshArgumentMap {
            {"filename", Filename}
      };

      // TODO get rid of this junk in favor of using WorldDirectiveMap

      const std::string AttributeBeginText = "AttributeBegin";
      const std::string WorldBeginText = "WorldBegin";

      bool IsQuoted(const std::string &token) {
         return (token[0] == '"' && token[token.size() - 1] == '"');
      }

      bool StartQuoted(const std::string &token) {
         return (token[0] == '"' && token[token.size() - 1] != '"');
      }

      bool EndQuoted(std::string token) {
         return (token[0] != '"' && token[token.size() - 1] == '"');
      }

      void LogOther(const PBRTDirective &directive, const std::string &error) {
         Log.WithTime(directive.Name + ": " + error);
      }

      void LogMissingArgument(const PBRTDirective &directive, const std::string& argument) {
         Log.WithTime("Directive [" + directive.Name + "] w/ identifier [" + directive.Identifier + "] is missing argument [" + argument + "]");
      }

      void LogMissingDirective(const std::string& name, std::string& defaultOption) {
         Log.WithTime("Directive [" + name + "] is missing, defaulting to " + defaultOption + ".");
      }

      void LogUnknownDirective(const PBRTDirective &directive) {
         Log.WithTime("Directive [" + directive.Name + "] found, but is unknown. Ignoring.");
      }

      void LogUnknownIdentifier(const PBRTDirective &directive) {
         Log.WithTime("Directive [" + directive.Name + "] has unknown identifier [" + directive.Identifier + "].");
      }

      void LogUnknownArgument(const PBRTArgument &argument) {
         Log.WithTime("Unknown argument type/name combination: [" + argument.Type + "] / [" + argument.Name + "].");
      }

      void LogWrongArgumentType(const PBRTDirective &directive, const PBRTArgument &argument) {
         Log.WithTime("Directive [" + directive.Name + "] w/ identifier [" + directive.Identifier + "] found has argument [" + argument.Name + "] with wrong type [" + argument.Type + "].");
      }

      void LogUnimplementedDirective(const PBRTDirective &directive) {
         Log.WithTime("Directive [" + directive.Name + "] w/ identifier [" + directive.Identifier + "] found, but is not yet implemented. Ignoring.");
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

      std::string unixifiedFilePath = filepath;

      std::string::size_type n = 0;
      while (( n = unixifiedFilePath.find(WindowsPathSeparator, n ) ) != std::string::npos) {
         unixifiedFilePath.replace( n, unixifiedFilePath.size(), UnixPathSeparator);
         n += unixifiedFilePath.size();
      }

      //std::replace(unixifiedFilePath.begin(), unixifiedFilePath.end(), WindowsPathSeparator, UnixPathSeparator);

      std::vector<std::vector<std::string>> tokens = Scan(OpenStream(filepath));

      // determine the name of the file from the given path
      const size_t lastPos = unixifiedFilePath.find_last_of(UnixPathSeparator);

      if (lastPos == std::string::npos) {
         _inputFilename = filepath;
         _basePathFromCWD = "";
      }
      else {
         _basePathFromCWD = filepath.substr(0, lastPos + 1);
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

            // check if this is a directive
            if (WorldDirectiveMap.count(word) > 0 || SceneDirectiveMap.count(word) > 0) {
            //if (std::find(Directives.begin(), Directives.end(), word) != Directives.end()) {
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
                  tokens[targetLineNumber].push_back(word.substr(0, lastIndex));
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

      bool missingSampler = true;

      unsigned int numSamples = DefaultSamples;

      for (const PBRTDirective& directive : sceneDirectives) {
         if (directive.Name == SamplerText) {
            missingSampler = false;
            if (directive.Identifier == "halton") {
               _sampler = std::make_unique<HaltonSampler>();
            } else {
               LogUnknownIdentifier(directive);
               _sampler = std::make_unique<HaltonSampler>();
            }

            for (const PBRTArgument& arg : directive.Arguments) {
               if (arg.Type == IntegerText) {
                  if (arg.Name == "pixelsamples") {
                     numSamples = stoui(arg.Values[0]);
                     break;
                  } else {
                     LogUnknownArgument(arg);
                  }
                  break;
               }
            }
            break;
         }
      }

      if (missingSampler) {
         std::string defaultOption = "Halton";
         LogMissingDirective(SamplerText, defaultOption);
      }

      if (_sampler == nullptr) {
         _sampler = std::make_unique<HaltonSampler>();
      }

      // film

      bool missingFilm = true;

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
                        LogUnknownArgument(arg);
                     }
                  } else if (arg.Type == StringText) {
                     if (arg.Name == "filename") {
                        filename = arg.Values[0];
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
         LogMissingDirective(FilmText, defaultOption);
      }

      if (_film == nullptr) {
         std::string filename = "polytope.png";
         _bounds.x = DefaultBoundsX;
         _bounds.y = DefaultBoundsY;
         _film = std::make_unique<PNGFilm>(_bounds, filename, std::move(_filter));
      }

      // filter

      bool missingFilter = true;

      for (const PBRTDirective& directive : sceneDirectives) {
         if (directive.Name == PixelFilterText) {
            missingFilter = false;
            if (directive.Identifier == "box") {
               unsigned int xWidth = 0;
               unsigned int yWidth = 0;

               for (const PBRTArgument& arg : directive.Arguments) {
                  if (arg.Type == IntegerText) {
                     if (arg.Name == "xwidth") {
                        xWidth = stoui(arg.Values[0]);
                     } else if (arg.Name == "ywidth") {
                        yWidth = stoui(arg.Values[0]);
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
         LogMissingDirective(PixelFilterText, defaultOption);
      }

      if (_filter == nullptr) {
         _filter = std::make_unique<BoxFilter>(_bounds);
         _film->Filter = std::move(_filter);
      }

      // camera

      std::unique_ptr<AbstractCamera> camera;

      {
         Transform currentTransform;

         for (const PBRTDirective& directive : sceneDirectives) {
            if (directive.Name == LookAtText) {
               if (directive.Arguments.size() == 1) {
                  if (directive.Arguments[0].Values.size() == 9) {
                     const float eyeX = stof(directive.Arguments[0].Values[0]);
                     const float eyeY = stof(directive.Arguments[0].Values[1]);
                     const float eyeZ = stof(directive.Arguments[0].Values[2]);

                     const Point eye = Point(eyeX, eyeY, eyeZ);

                     const float lookAtX = stof(directive.Arguments[0].Values[3]);
                     const float lookAtY = stof(directive.Arguments[0].Values[4]);
                     const float lookAtZ = stof(directive.Arguments[0].Values[5]);

                     const Point lookAt = Point(lookAtX, lookAtY, lookAtZ);

                     const float upX = stof(directive.Arguments[0].Values[6]);
                     const float upY = stof(directive.Arguments[0].Values[7]);
                     const float upZ = stof(directive.Arguments[0].Values[8]);

                     Vector up = Vector(upX, upY, upZ);

                     Transform t = Transform::LookAt(eye, lookAt, up);

                     currentTransform = currentTransform * t;
                  }
               }
               break;
            }
         }

         CameraSettings settings = CameraSettings(_bounds, DefaultCameraFOV);

         bool foundCamera = false;

         for (const PBRTDirective &directive : sceneDirectives) {
            if (directive.Name == CameraText) {
               if (directive.Identifier == "perspective") {
                  float fov = DefaultCameraFOV;

                  foundCamera = true;

                  for (const PBRTArgument& arg : directive.Arguments) {
                     if (arg.Type == FloatText) {
                        if (arg.Name == "fov") {
                           // TODO remove static_cast?
                           fov = static_cast<float>(stof(arg.Values[0]));
                        } else {
                           LogUnknownArgument(arg);
                        }
                     }
                  }

                  settings.FieldOfView = fov;

                  camera = std::make_unique<PerspectiveCamera>(settings, currentTransform, true);
                  break;
               }
            }
         }

         if (!foundCamera) {
            std::ostringstream stringstream;
            stringstream << "PerspectiveCamera with FOV = " << DefaultCameraFOV;
            std::string cameraDefaultString = stringstream.str();
            LogMissingDirective(CameraText, cameraDefaultString);
            camera = std::make_unique<PerspectiveCamera>(settings, currentTransform, true);
         }
      }

      // integrator

      bool missingIntegrator = true;

      for (const PBRTDirective& directive : sceneDirectives) {
         if (directive.Name == IntegratorText) {
            if (directive.Identifier == "path") {
               unsigned int maxDepth = 5;

               missingIntegrator = false;

               bool missingDepth = true;

               for (const PBRTArgument& arg : directive.Arguments) {
                  if (arg.Type == IntegerText) {
                     if (arg.Name == "maxdepth") {
                        maxDepth = stoui(arg.Values[0]);
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
               _integrator = std::make_unique<PathTraceIntegrator>(maxDepth);
            }
            else {
               LogUnimplementedDirective(directive);
            }
            break;
         }
      }

      if (missingIntegrator) {
         std::string text = "PathTraceIntegrator with MaxDepth = 5";
         LogMissingDirective(IntegratorText, text);
      }

      if (_integrator == nullptr) {
         _integrator = std::make_unique<PathTraceIntegrator>(5);
      }

      // world

      std::vector<std::shared_ptr<Polytope::Material>> namedMaterials;

      std::stack<std::shared_ptr<Polytope::Material>> materialStack;
      std::stack<std::shared_ptr<SpectralPowerDistribution>> lightStack;
      std::stack<std::shared_ptr<Transform>> transformStack;

      std::shared_ptr<Polytope::Material> activeMaterial;
      std::shared_ptr<SpectralPowerDistribution> activeLight;
      std::shared_ptr<Transform> activeTransform = std::make_shared<Transform>();

      const std::shared_ptr<Polytope::Material> materialMarker = std::make_shared<Polytope::Material>(AttributeBeginText);
      const std::shared_ptr<SpectralPowerDistribution> lightMarker = std::make_shared<SpectralPowerDistribution>();
      const std::shared_ptr<Transform> transformMarker = std::make_shared<Transform>();

      _scene = new NaiveScene(std::move(camera));

      for (const PBRTDirective& directive : worldDirectives) {
         WorldDirectiveName name;
         try {
            name = WorldDirectiveMap.at(directive.Name);
         }
         catch (...) {
            LogUnknownDirective(directive);
            continue;
         }

         switch (name) {
            case WorldDirectiveName::AreaLightSource: {
               // lights with geometry
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
               break;
            }
            case WorldDirectiveName::AttributeBegin: {
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
               break;
            }
            case WorldDirectiveName::AttributeEnd: {
               // pop material stack
               if (!materialStack.empty()) {
                  std::shared_ptr<Polytope::Material> stackValue = materialStack.top();
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
               break;
            }
            case WorldDirectiveName::LightSource: {
               // lights without geometry
               if (directive.Identifier == "infinite") {
                  for (const PBRTArgument& argument : directive.Arguments) {
                     if (argument.Name == "L") {
                        const float r = stof(argument.Values[0]);
                        const float g = stof(argument.Values[1]);
                        const float b = stof(argument.Values[2]);

                        const Polytope::SpectralPowerDistribution spd(r * 255, g * 255, b * 255);

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
            case WorldDirectiveName::MakeNamedMaterial: {
               const std::string materialName = directive.Identifier;
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

               namedMaterials.push_back(std::make_shared<Polytope::Material>(std::move(brdf), reflectanceSpectrum, materialName));
               break;
            }
            case WorldDirectiveName::Material: {
               MaterialIdentifier identifier;
               try {
                  identifier = MaterialIdentifierMap.at(directive.Identifier);
               }
               catch (...) {
                  LogUnknownIdentifier(directive);
                  continue;
               }
               switch (identifier) {
                  case MaterialIdentifier::Matte: {
                     for (const PBRTArgument& argument : directive.Arguments) {
                        if (argument.Name == "Kd" && argument.Type == "rgb") {
                           const float r = stof(argument.Values[0]);
                           const float g = stof(argument.Values[1]);
                           const float b = stof(argument.Values[2]);

                           ReflectanceSpectrum refl(r, g, b);
                           std::shared_ptr<Polytope::AbstractBRDF> brdf = std::make_shared<Polytope::LambertBRDF>();
                           std::shared_ptr<Polytope::Material> material = std::make_shared<Polytope::Material>(brdf, refl);
                           activeMaterial = material;
                        }
                     }
                  }
               }
               break;
            }
            case WorldDirectiveName::NamedMaterial: {
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
               break;
            }
            case WorldDirectiveName::Rotate: {
               // TODO need to ensure just 1 argument with 4 values
               PBRTArgument argument = directive.Arguments[0];
               const float angle = std::stof(argument.Values[0]) * PIOver180;
               float x = std::stof(argument.Values[1]);
               float y = std::stof(argument.Values[2]);
               float z = std::stof(argument.Values[3]);

               // normalize
               const float oneOverLength = 1.f / std::sqrt(x * x + y * y + z * z);
               x *= oneOverLength;
               y *= oneOverLength;
               z *= oneOverLength;

               // todo implement rotate...
               Transform t = Transform::Rotate(angle, x, y, z);

               if (activeTransform == nullptr) {
                  activeTransform = std::make_shared<Transform>();
               }

               Transform *active = activeTransform.get();
               *active = t * *active;
               break;
            }
            case WorldDirectiveName::Shape: {
               ShapeIdentifier identifier;
               try {
                  identifier = ShapeIdentifierMap.at(directive.Identifier);
               }
               catch (...) {
                  LogUnknownIdentifier(directive);
                  continue;
               }
               switch (identifier) {
                  case ShapeIdentifier::OBJMesh: {
                     // make sure it has a filename argument
                     bool filenameMissing = true;
                     std::string objFilename;
                     for (const PBRTArgument& argument : directive.Arguments) {
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
                              objFilename = argument.Values[0];
                              if (argument.Type != StringText) {
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
                        LogMissingArgument(directive, "filename");
                        break;
                     }

                     std::shared_ptr<Polytope::Transform> activeInverse = std::make_shared<Polytope::Transform>(activeTransform->Invert());

                     Polytope::TriangleMesh* mesh = new TriangleMesh(activeTransform, activeInverse, activeMaterial);

                     const OBJFileParser parser;
                     const std::string absoluteObjFilepath = GetCurrentWorkingDirectory() + UnixPathSeparator + _basePathFromCWD + objFilename;
                     parser.ParseFile(mesh, absoluteObjFilepath);
                     //mesh->ObjectToWorld = *activeTransform;
                     mesh->Material = activeMaterial;
                     _scene->Shapes.push_back(mesh);
                     break;
                  }
//                  case ShapeIdentifier::Sphere: {
//                     PBRTArgument argument = directive.Arguments[0];
//                     if (argument.Type == FloatText) {
//                        float radius = std::stof(argument.Values[0]);
//                        AbstractShape *sphere = new Polytope::Sphere(*activeTransform, activeMaterial);
//                        if (activeLight != nullptr) {
//                           ShapeLight *sphereLight = new ShapeLight(*activeLight);
//                           sphere->Light = sphereLight;
//                           _scene->Lights.push_back(sphereLight);
//                        }
//                        _scene->Shapes.push_back(sphere);
//                     }
//                  }
                  default: {
                     LogUnimplementedDirective(directive);
                     break;
                  }
               }
               break;
            }
            case WorldDirectiveName::TransformBegin: {
               // push onto transform stack
               transformStack.push(transformMarker);
               if (activeTransform != nullptr) {
                  transformStack.push(activeTransform);
               }
               break;
            }
            case (WorldDirectiveName::TransformEnd): {
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
               break;
            }
            case WorldDirectiveName::Translate: {
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
         std::move(_sampler),
         _scene,
         std::move(_integrator),
         std::move(_film),
         _bounds,
         numSamples
      );
   }
}
