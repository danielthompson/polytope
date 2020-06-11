//
// Created by Daniel on 15-Dec-19.
//

#ifndef POLY_ABSTRACTFILEPARSER_H
#define POLY_ABSTRACTFILEPARSER_H

#include <fstream>
#include <memory>

namespace poly {
   class AbstractFileParser {
   public:
      std::unique_ptr<std::ifstream> open_ascii_stream(const std::string &filepath) const;
      std::unique_ptr<std::ifstream> open_binary_stream(const std::string &filepath) const;
      unsigned int stoui(const std::string& text) const;
   private:
      std::unique_ptr<std::ifstream> open_stream(const std::string &filepath, const bool binary) const;
   };
}

#endif //POLY_ABSTRACTFILEPARSER_H
