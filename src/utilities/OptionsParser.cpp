//
// Created by Daniel on 29-Mar-18.
//

#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include "OptionsParser.h"

namespace Polytope {

   OptionsParser::OptionsParser(int &argc, char **argv) {
      for (int i=1; i < argc; ++i)
         this->_tokens.emplace_back(argv[i]);
   }

   const std::string& OptionsParser::GetOption(const std::string &option) const {
      std::vector<std::string>::const_iterator itr;

      itr = std::find(std::begin(_tokens), std::end(_tokens), option);
      if (itr != this->_tokens.end() && ++itr != this->_tokens.end()){
         return *itr;
      }
      static const std::string empty_string("");
      return empty_string;
   }

   bool OptionsParser::OptionExists(const std::string &option) const {
      return std::find(this->_tokens.begin(), this->_tokens.end(), option)
             != this->_tokens.end();
   }

   Polytope::Options OptionsParser::Parse() {
      Polytope::Options options = Polytope::Options();

      if (OptionExists("--help")) {
         std::cout << "Polytope help text blah blah blah" << std::endl;
      }

      if (OptionExists("-threads")) {
         const std::string threads = GetOption("-threads");

         try {
            options.threads = std::stoi(threads);
            std::cout << "Parsed \"-threads " << threads << "\", using " << options.threads << " threads." << std::endl;
         }
         catch (std::invalid_argument&) {
            options.valid = false;
            std::cout << "Failed to parse \"-threads " << threads << "\"." << std::endl;
         }
      }

      return options;
   }



}
