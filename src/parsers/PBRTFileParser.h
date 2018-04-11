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

      const std::vector<std::string> Directives {
         "AreaLightSource",
         "AttributeBegin", // done
         "AttributeEnd", // done
         "Camera",
         "Film",
         "Integrator",
         "LookAt", // done
         "MakeNamedMaterial",
         "NamedMaterial", // done
         "PixelFilter",
         "Sampler",
         "Shape",
         "TransformBegin", // done
         "TransformEnd", // done
         "Translate", // done
         "WorldBegin", // done
         "WorldEnd" // done
      };

   };

}


#endif //POLYTOPE_FILEPARSER_H
