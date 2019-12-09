//
// Created by Daniel on 29-Mar-18.
//

#ifndef POLYTOPE_OPTIONSPARSER_H
#define POLYTOPE_OPTIONSPARSER_H

#include "Options.h"
#include "Logger.h"

namespace Polytope {

   class OptionsParser {
   public:
      OptionsParser(int &argc, char **argv, Polytope::Logger &logger);
      void Parse(Polytope::Options &options) const;

      Polytope::Logger Logger;

   private:
      std::vector<std::string> _tokens;
      bool OptionExists(const std::string &option) const;
      const std::string &GetOption(const std::string &option) const;

      unsigned int stou(const std::string &str, size_t *idx = nullptr, int base = 10) const ;
   };

}


#endif //POLYTOPE_OPTIONSPARSER_H
