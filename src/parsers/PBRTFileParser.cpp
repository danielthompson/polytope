//
// Created by Daniel on 07-Apr-18.
//

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include "PBRTFileParser.h"
#include "../integrators/PathTraceIntegrator.h"
#include "../samplers/HaltonSampler.h"
#include "../runners/TileRunner.h"

namespace Polytope {

   std::unique_ptr<AbstractRunner> PBRTFileParser::Parse() noexcept(false){

      std::ifstream file(Filename);

      std::vector<std::vector<std::string>> tokens;

      // scan

      if (file.is_open())
      {
         int sourceLineNumber = 0;
         int targetLineNumber = -1;
         std::string line;
         while (getline(file, line))
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
         file.close();
      }
      else {
         throw std::invalid_argument("Couldn't open file " + Filename);
      }

      // parse

      std::vector<PBRTDirective> directives;

      for (std::vector<std::string> line : tokens) {
         PBRTDirective currentDirective = PBRTDirective();

         if (line.empty())
            continue;

         currentDirective.Name = line[0];

         if (line.size() == 1) {
            //currentDirective.
            directives.push_back(currentDirective);

            continue;
         }

         if (IsQuoted(line[1])) {
            currentDirective.Identifier = line[1].substr(1, line[1].length() - 2);
         }
         else {
            currentDirective.Arguments = std::vector<PBRTArgument>();
            PBRTArgument argument = PBRTArgument();
            argument.Type = "float";
            argument.Values = std::vector<std::string>();

            for (int i = 1; i < line.size(); i++) {
               argument.Values.push_back(line[i]);
            }

            currentDirective.Arguments.push_back(argument);
            directives.push_back(currentDirective);
            continue;
         }

         if (line.size() == 2)
            continue;

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
               }
               else {
                  currentArgument.Values.push_back(line[i]);
               }
               i++;
               continue;
            }
         }

         if (inValue) {
            currentDirective.Arguments.push_back(currentArgument);
         }

         directives.push_back(currentDirective);
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