//
// Created by Daniel on 21-Mar-18.
//

#ifndef POLY_LOGGER_H
#define POLY_LOGGER_H

#include <string>

namespace poly {

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


#endif //POLY_LOGGER_H
