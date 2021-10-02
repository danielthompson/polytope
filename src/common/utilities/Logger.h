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

#define LOG_LEVEL LOG_LEVEL_DEBUG 


#include "../../cpu/structures/Vectors.h"
#include "../../cpu/structures/ray.h"
#include <string>
#include <iostream>

namespace poly {
   class Logger {
   public:
      static std::ostream &debug();
      static std::ostream &info();
      static std::ostream &warning();
      static std::ostream &error();
   };
}

inline std::ostream& operator << (std::ostream &os, const poly::point &p) {
   return (os << "(" << p.x << ", " << p.y << ", " << p.z << ")");
}

inline std::ostream& operator << (std::ostream &os, const poly::vector &v) {
   return (os << "(" << v.x << ", " << v.y << ", " << v.z << ")");
}

inline std::ostream& operator << (std::ostream &os, const poly::ray &r) {
   return (os << "o: " << r.origin << ", d: " << r.direction);
}

#if LOG_LEVEL <= LOG_LEVEL_DEBUG
#define LOG_DEBUG(...) do { Log.debug() << __VA_ARGS__ ; } while (0)
#else
#define LOG_DEBUG(...) do { ; } while (0)
#endif

#if LOG_LEVEL <= LOG_LEVEL_INFO
#define LOG_INFO(...) do { Log.info() << __VA_ARGS__ ; } while (0)
#else
#define LOG_INFO(...) do { ; } while (0)
#endif

#if LOG_LEVEL <= LOG_LEVEL_WARNING
#define LOG_WARNING(...) do { Log.warning() << __VA_ARGS__ ; } while (0)
#else
#define LOG_WARNING(...) do { ; } while (0)
#endif

#if LOG_LEVEL <= LOG_LEVEL_ERROR
#define LOG_ERROR(...) do { Log.error() << __VA_ARGS__ ; } while (0)
#else
#define LOG_ERROR(...) do { ; } while (0)
#endif


#endif //POLY_LOGGER_H
