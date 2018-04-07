//
// Created by Daniel on 07-Apr-18.
//

#include <fstream>
#include <sstream>
#include "FileParser.h"
#include "../integrators/PathTraceIntegrator.h"

namespace Polytope {

   std::unique_ptr<AbstractRunner> FileParser::Parse() noexcept(false){
      std::string line;
      std::ifstream file(Filename);

      std::vector<std::vector<std::string>> tokens;

      if (file.is_open())
      {
         int lineNumber = 0;
         while (getline(file, line))
         {
            tokens.emplace_back();
            std::string word;
            std::istringstream iss(line, std::istringstream::in);
            while (iss >> word)
            {
               tokens[lineNumber].push_back(word);
            }

            lineNumber++;
         }
         file.close();
      }
      else {
         throw std::invalid_argument("Couldn't open file " + Filename);
      }

      return std::unique_ptr<AbstractRunner>();
   }
}