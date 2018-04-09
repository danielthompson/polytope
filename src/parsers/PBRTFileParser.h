//
// Created by Daniel on 07-Apr-18.
//

#ifndef POLYTOPE_FILEPARSER_H
#define POLYTOPE_FILEPARSER_H

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

   class PBRTFileParser {
   public:

      // constructors
      explicit PBRTFileParser(const std::string &filename) : Filename(filename) { };

      // methods
      std::unique_ptr<AbstractRunner> Parse() noexcept(false);

      std::string Filename;

      std::unique_ptr<AbstractRunner> Runner;
   private:

      bool IsQuoted(std::string token);
      bool StartQuoted(std::string token);
      bool EndQuoted(std::string token);

      void CreateSampler(std::vector<std::string> &directive);
      void CreateIntegrator(std::vector<std::string> &directive);

      std::unique_ptr<AbstractSampler> Sampler;
      AbstractScene *scene;
      std::unique_ptr<AbstractIntegrator> Integrator;
      std::unique_ptr<AbstractFilm> Film;
      unsigned int numSamples;
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
