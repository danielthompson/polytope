//
// Created by Daniel on 15-Dec-19.
//

#include <cstring>
#include "AbstractFileParser.h"
#include "../utilities/Common.h"
#include "errno.h"

namespace Polytope {
   std::unique_ptr<std::istream> AbstractFileParser::OpenStream(const std::string &filepath) const {
      
      std::string absolute_path;
      if (filepath[0] != '/' && filepath[0] != '~') {
         // relative path was provided
         std::string cwd = GetCurrentWorkingDirectory();
         absolute_path = cwd + UnixPathSeparator + filepath;
      } 
      else {
         absolute_path = filepath;
      }

      Log.WithTime("Trying to open [" + absolute_path + "].");
      std::unique_ptr<std::istream> stream = std::make_unique<std::ifstream>(absolute_path);

      if (stream->good()) {
         Log.WithTime("Opened stream on [" + absolute_path + "].");

      } else {
         Log.WithTime("Couldn't open stream on [" + absolute_path + "] due to [" + strerror(errno));
         throw std::invalid_argument("Couldn't open stream on [" + absolute_path + "].");
      }

      return stream;
   }

   unsigned int AbstractFileParser::stoui(const std::string &text) const {
      return static_cast<unsigned int>(stoi(text));
   }
}
