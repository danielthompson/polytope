//
// Created by Daniel on 15-Dec-19.
//

#include "AbstractFileParser.h"
#include "../utilities/Common.h"

namespace Polytope {
   std::unique_ptr<std::istream> AbstractFileParser::OpenStream(const std::string &filepath) const {
      std::string cwd = GetCurrentWorkingDirectory();
      std::string absolutePath = cwd + "//" + filepath;

      Log.WithTime("Trying to open [" + absolutePath + "].");
      std::unique_ptr<std::istream> stream = std::make_unique<std::ifstream>(filepath);

      if (stream->good()) {
         Log.WithTime("Opened stream on [" + absolutePath + "].");

      } else {
         throw std::invalid_argument("Couldn't open stream on [" + absolutePath + "].");
      }

      return stream;
   }

   unsigned int AbstractFileParser::stoui(const std::string &text) const {
      return static_cast<unsigned int>(stoi(text));
   }
}
