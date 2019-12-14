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

      std::unique_ptr<AbstractRunner> ParseFile(const std::string &filepath) noexcept(false);
      std::unique_ptr<AbstractRunner> ParseString(const std::string &text) noexcept(false);



      std::unique_ptr<AbstractRunner> Runner;
   private:

      std::unique_ptr<AbstractRunner> Parse(std::vector<std::vector<std::string>> tokens) noexcept(false);
      std::vector<std::vector<std::string>> Scan(std::unique_ptr<std::istream> stream);

      bool IsQuoted(std::string token);
      bool StartQuoted(std::string token);
      bool EndQuoted(std::string token);
      void LogBadIdentifier(const PBRTDirective &directive);
      void LogOther(const PBRTDirective &directive, const std::string &error);

      void CreateSampler(std::vector<std::string> &directive);
      void CreateIntegrator(std::vector<std::string> &directive);

      std::unique_ptr<AbstractSampler> _sampler;
      AbstractScene *_scene = nullptr;
      std::unique_ptr<AbstractIntegrator> _integrator;
      std::unique_ptr<AbstractFilm> _film;
      std::unique_ptr<AbstractFilter> _filter;
      unsigned int _numSamples = 0;
      Polytope::Bounds _bounds;
   };
}

#endif //POLYTOPE_FILEPARSER_H
