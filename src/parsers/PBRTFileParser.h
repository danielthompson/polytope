//
// Created by Daniel on 07-Apr-18.
//

#ifndef POLYTOPE_FILEPARSER_H
#define POLYTOPE_FILEPARSER_H

#include <iostream>
#include <string>
#include "../runners/AbstractRunner.h"
#include "../utilities/Logger.h"

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
      explicit PBRTFileParser(const Polytope::Logger logger)
            : Logger(logger) { };

      std::unique_ptr<AbstractRunner> ParseFile(const std::string &filename) noexcept(false);
      std::unique_ptr<AbstractRunner> ParseString(const std::string &text) noexcept(false);

      std::string Filename;

      std::unique_ptr<AbstractRunner> Runner;
   private:

      std::unique_ptr<AbstractRunner> Parse(std::unique_ptr<std::istream> stream) noexcept(false);


      Polytope::Logger Logger;

      bool IsQuoted(std::string token);
      bool StartQuoted(std::string token);
      bool EndQuoted(std::string token);
      void LogBadArgument(const PBRTArgument &argument);
      void LogBadIdentifier(const PBRTDirective &directive);

      void CreateSampler(std::vector<std::string> &directive);
      void CreateIntegrator(std::vector<std::string> &directive);

      std::unique_ptr<AbstractSampler> Sampler;
      AbstractScene *scene = nullptr;
      std::unique_ptr<AbstractIntegrator> Integrator;
      std::unique_ptr<AbstractFilm> Film;
      std::unique_ptr<AbstractFilter> Filter;
      unsigned int numSamples = 0;
      Polytope::Bounds Bounds;

      // datatypes

      constexpr std::string IntegerText = "integer";
      constexpr std::string FloatText = "float";
      constexpr std::string StringText = "string";
      constexpr std::string RGBText = "rgb";

      // scene

      constexpr std::string CameraText = "Camera";
      constexpr std::string FilmText = "Film";
      constexpr std::string IntegratorText = "Integrator";
      constexpr std::string LookAtText = "LookAt";
      constexpr std::string PixelFilterText = "PixelFilter";
      constexpr std::string SamplerText = "Sampler";

      // world

      constexpr std::string AreaLightSourceText = "AreaLightSource";
      constexpr std::string AttributeBeginText = "AttributeBegin";
      constexpr std::string AttributeEndText = "AttributeEnd";
      constexpr std::string MakeNamedMaterialText = "MakeNamedMaterial";
      constexpr std::string NamedMaterialText = "NamedMaterial";
      constexpr std::string ShapeText = "Shape";
      constexpr std::string TransformBeginText = "TransformBegin";
      constexpr std::string TransformEndText = "TransformEnd";
      constexpr std::string TranslateText = "Translate";
      constexpr std::string WorldBeginText = "WorldBegin";
      constexpr std::string WorldEndText = "WorldEnd";

      constexpr std::vector<std::string> Directives {
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
