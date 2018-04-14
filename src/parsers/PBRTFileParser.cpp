//
// Created by Daniel on 07-Apr-18.
//

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stack>
#include "PBRTFileParser.h"
#include "../integrators/PathTraceIntegrator.h"
#include "../samplers/HaltonSampler.h"
#include "../runners/TileRunner.h"
#include "../films/PNGFilm.h"
#include "../filters/BoxFilter.h"
#include "../cameras/PerspectiveCamera.h"
#include "../shading/brdf/LambertBRDF.h"

namespace Polytope {

   std::unique_ptr<AbstractRunner> PBRTFileParser::ParseFile(const std::string &filename) {
      return Parse(std::make_unique<std::ifstream>(filename));
   }

   std::unique_ptr<AbstractRunner> PBRTFileParser::ParseString(const std::string &text) {
      return Parse(std::make_unique<std::istringstream>(text));
   }

   std::unique_ptr<AbstractRunner> PBRTFileParser::Parse(std::unique_ptr<std::istream> stream) noexcept(false){

      std::vector<std::vector<std::string>> tokens;

      // scan

      if (stream->good())
      {
         int sourceLineNumber = 0;
         int targetLineNumber = -1;
         std::string line;
         while (getline(*stream, line))
         {
            tokens.emplace_back();
            std::string word;
            std::istringstream iss(line, std::istringstream::in);
            while (iss >> word)
            {
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
                  }
                  else if (word[lastIndex] == ']') {
                     tokens[targetLineNumber].push_back(word.substr(0, lastIndex - 1));
                     tokens[targetLineNumber].push_back("]");
                  }
                  else {
                     tokens[targetLineNumber].push_back(word);
                  }
               }
               else {
                  tokens[targetLineNumber].push_back(word);
               }

            }

            sourceLineNumber++;

         }

      }
      else {
         throw std::invalid_argument("Couldn't open file " + Filename);
      }

      // parse

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

      for (PBRTDirective directive : sceneDirectives) {
         if (directive.Name == SamplerText) {
            if (directive.Identifier == "halton") {
               Sampler = std::make_unique<HaltonSampler>();
               break;
            } else {
               Logger.Log("Sampler identifier specified [" + directive.Identifier + "] is unknown, using Halton");
               Sampler = std::make_unique<HaltonSampler>();
            }

            for (PBRTArgument arg : directive.Arguments) {
               if (arg.Type == IntegerText) {
                  if (arg.Name == "pixelsamples") {
                     numSamples = static_cast<unsigned int>(stoi(arg.Values[0]));
                     break;
                  }
                  else {
                     LogBadArgument(arg);
                  }
               }
            }
         }
      }

      if (Sampler == nullptr) {
         Logger.Log("No Sampler specified, using Halton.");
         Sampler = std::make_unique<HaltonSampler>();
      }

      // filter

      bool createBoxFilter = false;

      for (PBRTDirective directive : sceneDirectives) {
         if (directive.Name == PixelFilterText) {
            if (directive.Identifier == "box") {
               createBoxFilter = true;
               unsigned int xWidth = 0;
               unsigned int yWidth = 0;

               for (PBRTArgument arg : directive.Arguments) {
                  if (arg.Type == IntegerText) {
                     if (arg.Name == "xwidth") {
                        xWidth = static_cast<unsigned int>(stoi(arg.Values[0]));
                     }
                     else if (arg.Name == "ywidth") {
                        yWidth = static_cast<unsigned int>(stoi(arg.Values[0]));
                     }
                     else {
                        LogBadArgument(arg);
                     }
                  }
               }
            }
            else {
               LogBadIdentifier(directive);
            }
         }
      }

      // film

      Polytope::Bounds bounds;

      for (PBRTDirective directive : sceneDirectives) {
         if (directive.Name == FilmText) {
            if (directive.Identifier == "image") {
               unsigned int x = 0;
               unsigned int y = 0;
               std::string filename;

               for (PBRTArgument arg : directive.Arguments) {
                  if (arg.Type == IntegerText) {
                     if (arg.Name == "xresolution") {
                        x = static_cast<unsigned int>(stoi(arg.Values[0]));
                     }
                     else if (arg.Name == "yresolution") {
                        y = static_cast<unsigned int>(stoi(arg.Values[0]));
                     }
                     else {
                        LogBadArgument(arg);
                     }
                  }
                  else if (arg.Type == StringText) {
                     if (arg.Name == "filename") {
                        filename = arg.Values[0];
                     }
                     else {
                        LogBadArgument(arg);
                     }
                  }
               }

               bounds = Polytope::Bounds(x, y);

               // TODO

               if (createBoxFilter) {
                  Filter = std::make_unique<BoxFilter>(bounds);
               }
               Film = std::make_unique<PNGFilm>(bounds, filename, std::move(Filter));
            }
         }
      }

      // camera

      Transform currentTransform;

      for (PBRTDirective directive : sceneDirectives) {
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

      std::unique_ptr<AbstractCamera> camera;

      CameraSettings settings = CameraSettings(bounds, 50);

      for (const PBRTDirective &directive : sceneDirectives) {
         if (directive.Name == CameraText) {
            if (directive.Identifier == "perspective") {
               float fov = 50;

               for (PBRTArgument arg : directive.Arguments) {
                  if (arg.Type == FloatText) {
                     if (arg.Name == "fov") {
                        fov = static_cast<float>(stof(arg.Values[0]));
                     }
                     else {
                        LogBadArgument(arg);
                     }
                  }
               }

               settings.FieldOfView = fov;

               camera = std::make_unique<PerspectiveCamera>(settings, currentTransform);
            }
         }
      }

      // world

      std::vector<Material> namedMaterials;

      std::stack<Transform> transformStack;
      std::stack<Material> materialStack;

      Material currentMaterial;

      for (const PBRTDirective &directive : worldDirectives) {
         if (directive.Name == MakeNamedMaterialText) {
            // how to make a named material?
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

            Material material = Material(std::move(brdf), reflectanceSpectrum);
            material.Name = materialName;

            namedMaterials.push_back(material);

         }

         if (directive.Name == AttributeBeginText) {

         }

      }

      return std::make_unique<TileRunner>(
            std::move(Sampler),
            scene,
            std::move(Integrator),
            std::move(Film),
            Bounds,
            numSamples
      );
   }

   void PBRTFileParser::LogBadArgument(const PBRTArgument &argument) {
      Logger.Log("Unknown argument type/name combination: [" + argument.Type + "] / [" + argument.Name + "].");
   }

   void PBRTFileParser::LogBadIdentifier(const PBRTDirective &directive) {
      Logger.Log("Unknown directive/identifier combination: [" + directive.Name + "] / [" + directive.Identifier + "].");
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
         Sampler = std::make_unique<HaltonSampler>();
      }
      else {
         throw std::invalid_argument("Given sampler [" + directive[1] + "] not supported");
      }

      numSamples = std::stoi(directive[4]);
   }

   void PBRTFileParser::CreateIntegrator(std::vector<std::string> &directive) {
      if (directive[1] == "\"path\"") {
         Sampler = std::make_unique<HaltonSampler>();
      }
      else {
         throw std::invalid_argument("Given sampler [" + directive[1] + "] not supported");
      }

      numSamples = std::stoi(directive[4]);
   }




}