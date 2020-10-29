//
// Created by Daniel on 21-Mar-18.
//

#ifndef POLY_LOGGER_H
#define POLY_LOGGER_H

#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_WARNING 2
#define LOG_LEVEL_ERROR 3
#define LOG_LEVEL_NONE 4

#define LOG_LEVEL LOG_LEVEL_INFO 

#include <string>

namespace poly {

   class Logger {
   public:
      
      Logger() = default;

      void debug(const char* format, ...);
      void info(const char* format, ...);
      void warning(const char* format, ...);
      void error(const char* format, ...);

      void debug(const std::string& text);
      void info(const std::string& text);
      void warning(const std::string& text);
      void error(const std::string& text);
      
//      void logThread(const std::string& text) const;

   private:
      
      enum log_level {
         LOG_DEBUG,
         LOG_INFO,
         LOG_WARNING,
         LOG_ERROR,
      };
      void log(Logger::log_level level, const char *format, va_list args);
      void log(log_level level, const std::string& text);
   };
}


#endif //POLY_LOGGER_H
