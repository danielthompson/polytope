//
// Created by Daniel on 15-Dec-19.
//

#include <cstring>
#include "AbstractFileParser.h"
#include "../utilities/Common.h"
#include "errno.h"

namespace poly {

   std::unique_ptr<std::ifstream> AbstractFileParser::open_stream(const std::string &filepath, const bool binary) const {
      std::string absolute_path;
      if (filepath[0] != '/' && filepath[0] != '~') {
         // relative path was provided
         std::string cwd = GetCurrentWorkingDirectory();
         absolute_path = cwd + UnixPathSeparator + filepath;
      }
      else {
         absolute_path = filepath;
      }

      std::string mode_string = "ascii";
      std::ios_base::openmode mode = std::ios_base::in;
      
      if (binary) {
         mode = std::ios_base::in | std::ios_base::binary;
         mode_string = "binary";
      }
      
      Log.WithTime("Trying to open [" + absolute_path + "].");
      std::unique_ptr<std::ifstream> stream = std::make_unique<std::ifstream>(absolute_path, mode);

      if (stream->good()) {
         Log.WithTime("Opened " + mode_string + " stream on [" + absolute_path + "].");
      } else {
         ERROR("Couldn't open stream on [%s] due to [%s]", absolute_path.c_str(), strerror(errno));
      }

      return stream;
   }
   
   std::unique_ptr<std::ifstream> AbstractFileParser::open_ascii_stream(const std::string &filepath) const {
      return open_stream(filepath, false);
   }

   std::unique_ptr<std::ifstream> AbstractFileParser::open_binary_stream(const std::string &filepath) const {
      return open_stream(filepath, true);
   }

   unsigned int AbstractFileParser::stoui(const std::string &text) const {
      return static_cast<unsigned int>(stoi(text));
   }
}
