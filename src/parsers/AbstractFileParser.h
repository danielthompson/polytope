//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLYTOPE_ABSTRACTFILEPARSER_H
#define POLYTOPE_ABSTRACTFILEPARSER_H

#include <fstream>
#include <memory>

namespace Polytope {
   class AbstractFileParser {
   public:
      std::unique_ptr<std::istream> OpenStream(const std::string &filepath) const;
      unsigned int stoui(const std::string& text) const;
   };
}

#endif //POLYTOPE_ABSTRACTFILEPARSER_H
