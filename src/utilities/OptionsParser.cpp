//
// Created by Daniel on 29-Mar-18.
//

#include <cstring>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include "OptionsParser.h"
#include "Common.h"

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
         options.help = true;
         return options;
      }

      Log.WithTime("Polytope started.");

      std::string option = "-threads";

      if (OptionExists(option)) {
         const std::string value = GetOption(option);

         try {
            unsigned int parsedValue = stou(value);
            options.threads = parsedValue;
            Log.WithTime("Parsed [" + option + "] = [" + std::to_string(parsedValue) + "].");
         }
         catch (...) {
            options.valid = false;
            Log.WithTime("Failed to parse [" + option + "] = [" + value + "].");
         }
      }

      option = "-samples";

      if (OptionExists(option)) {
         const std::string value = GetOption(option);

         try {
            unsigned int parsedValue = stou(value);
            options.samples = parsedValue;
            Log.WithTime("Parsed [" + option + "] = [" + std::to_string(parsedValue) + "].");
         }
         catch (...) {
            options.valid = false;
            Log.WithTime("Failed to parse [" + option + "] = [" + value + "].");
         }
      }

      option = "-inputfile";

      if (OptionExists(option)) {
         const std::string value = GetOption(option);

         try {
            options.input_filename = value;
            Log.WithTime("Parsed [" + option + "] = [" + value + "].");
         }
         catch (...) {
            options.valid = false;
            Log.WithTime("Failed to parse [" + option + "] = [" + value + "].");
         }
      }

      option = "-outputfile";

      if (OptionExists(option)) {
         const std::string value = GetOption(option);

         try {
            options.output_filename = value;
            Log.WithTime("Parsed [" + option + "] = [" + value + "].");
         }
         catch (...) {
            options.valid = false;
            Log.WithTime("Failed to parse [" + option + "] = [" + value + "].");
         }
      }

      return options;

   }

   unsigned int OptionsParser::stou(std::string const &str, size_t *idx, int base) const {
      unsigned long result = std::stoul(str, idx, base);
      if (result > std::numeric_limits<unsigned>::max()) {
         throw std::out_of_range("stou");
      }
      return result;
   }
}
