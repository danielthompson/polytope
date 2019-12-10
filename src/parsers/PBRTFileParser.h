//
// Created by Daniel on 07-Apr-18.
//

#ifndef POLYTOPE_FILEPARSER_H
#define POLYTOPE_FILEPARSER_H

#include <iostream>
#include <string>
#include "../runners/AbstractRunner.h"

namespace Polytope {

   class PBRTArgument {
   public:
      std::string Type;
      std::string Name;
      std::vector<std::string> Values;
   };

   class PBRTDirective {
   public:
      std::string Name;
      std::string Identifier;
      std::vector<Polytope::PBRTArgument> Arguments;
   };

   class PBRTGraphicsState {
   public:
      std::unique_ptr<Material> material;

   };

   class PBRTFileParser {
   public:

      // constructors
      explicit PBRTFileParser() = default;

      std::unique_ptr<AbstractRunner> ParseFile(const std::string &filename) noexcept(false);
      std::unique_ptr<AbstractRunner> ParseString(const std::string &text) noexcept(false);

      std::string Filename;

      std::unique_ptr<AbstractRunner> Runner;
   private:

      std::unique_ptr<AbstractRunner> Parse(std::vector<std::vector<std::string>> tokens) noexcept(false);
      std::vector<std::vector<std::string>> Scan(std::unique_ptr<std::istream> stream);

      bool IsQuoted(std::string token);
      bool StartQuoted(std::string token);
      bool EndQuoted(std::string token);
      void LogBadArgument(const PBRTArgument &argument);
      void LogBadIdentifier(const PBRTDirective &directive);
      void LogOther(const PBRTDirective &directive, const std::string &error);

      void CreateSampler(std::vector<std::string> &directive);
      void CreateIntegrator(std::vector<std::string> &directive);

      std::unique_ptr<AbstractSampler> Sampler;
      AbstractScene *Scene = nullptr;
      std::unique_ptr<AbstractIntegrator> Integrator;
      std::unique_ptr<AbstractFilm> Film;
      std::unique_ptr<AbstractFilter> Filter;
      unsigned int numSamples = 0;
      Polytope::Bounds Bounds;

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
   };
}

#endif //POLYTOPE_FILEPARSER_H
