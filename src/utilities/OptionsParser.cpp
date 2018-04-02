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

   OptionsParser::OptionsParser(int &argc, char **argv, Polytope::Logger &logger)
   : Logger(logger) {
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

      std::string option = "-threads";

      if (OptionExists(option)) {
         const std::string value = GetOption(option);

         try {
            unsigned int parsedValue = stou(value);
            options.threads = parsedValue;
            Logger.LogTime("Parsed [" + option + "] = [" + std::to_string(parsedValue) + "].");
         }
         catch (...) {
            options.valid = false;
            Logger.LogTime("Failed to parse [" + option + "] = [" + value + "].");
         }
      }

      option = "-samples";

      if (OptionExists(option)) {
         const std::string value = GetOption(option);

         try {
            unsigned int parsedValue = stou(value);
            options.samples = parsedValue;
            Logger.LogTime("Parsed [" + option + "] = [" + std::to_string(parsedValue) + "].");
         }
         catch (...) {
            options.valid = false;
            Logger.LogTime("Failed to parse [" + option + "] = [" + value + "].");
         }
      }

      return options;
   }

   unsigned int OptionsParser::stou(std::string const & str, size_t * idx, int base)  {
      unsigned long result = std::stoul(str, idx, base);
      if (result > std::numeric_limits<unsigned>::max()) {
         throw std::out_of_range("stou");
      }
      return result;
   }


}
