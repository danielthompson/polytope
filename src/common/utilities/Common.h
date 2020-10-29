//
// Created by Daniel on 09-Dec-19.
//

#ifndef POLY_COMMON_H
#define POLY_COMMON_H

#include <atomic>
#include "Logger.h"

extern poly::Logger Log;

#define ERROR(fmt, ...) do { Log.error(fmt, ##__VA_ARGS__); exit(1); } while (0)

std::string GetCurrentWorkingDirectory();

inline std::string add_commas(size_t number) {
   auto s = std::to_string(number);
   int n = s.length() - 3;
   while (n > 0) {
      s.insert(n, ",");
      n -= 3;
   }
   return s;
}

extern std::string WindowsPathSeparator;
extern std::string UnixPathSeparator;

#endif //POLY_COMMON_H
