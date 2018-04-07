//
// Created by Daniel on 07-Apr-18.
//

#ifndef POLYTOPE_FILEPARSER_H
#define POLYTOPE_FILEPARSER_H

#include <string>
#include "../runners/AbstractRunner.h"

namespace Polytope {

   class FileParser {
   public:

      // constructors
      explicit FileParser(std::string &filename) : Filename(filename) { };

      // methods
      std::unique_ptr<AbstractRunner> Parse() noexcept(false);

      std::string Filename;
   };

}


#endif //POLYTOPE_FILEPARSER_H
