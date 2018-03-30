//
// Created by Daniel on 21-Mar-18.
//

#ifndef POLYTOPE_LOGGER_H
#define POLYTOPE_LOGGER_H

#include <string>
#include <map>

namespace Polytope {

   class Logger {

   public:

      Logger() = default;
      //explicit Logger(std::map<std::thread::id, int> threadMap) : threadMap(threadMap) { }

      std::string time_in_HH_MM_SS_MMM() const;
      void LogTime(std::string text) const;
      void Log(std::string text) const;
      void logThread(std::string text) const;
   };
}


#endif //POLYTOPE_LOGGER_H
