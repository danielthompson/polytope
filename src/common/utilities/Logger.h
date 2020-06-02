//
// Created by Daniel on 21-Mar-18.
//

#ifndef POLYTOPE_LOGGER_H
#define POLYTOPE_LOGGER_H

#include <string>

namespace Polytope {

   class Logger {
   public:

      Logger() = default;
      void ErrorWithTime(const std::string& text) const;
      void WithTime(const std::string& text) const;
      void Log(const std::string& text) const;
      void logThread(const std::string& text) const;

   private:
      std::string time_in_HH_MM_SS_MMM() const;
      void WithTime(const std::ostream out, const std::string& text) const;
   };
}


#endif //POLYTOPE_LOGGER_H