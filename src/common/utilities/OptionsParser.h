//
// Created by Daniel on 29-Mar-18.
//

#ifndef POLY_OPTIONSPARSER_H
#define POLY_OPTIONSPARSER_H

#include <vector>
#include "Options.h"

namespace poly {

   class OptionsParser {
   public:
      OptionsParser(int &argc, char **argv);
      poly::Options Parse();

   private:
      std::vector<std::string> _tokens;
      bool OptionExists(const std::string &option) const;
      const std::string &GetOption(const std::string &option) const;

      unsigned int stou(const std::string &str, size_t *idx = nullptr, int base = 10) const ;
   };

}


#endif //POLY_OPTIONSPARSER_H
