//
// Created by Daniel on 09-Dec-19.
//

#ifndef POLY_COMMON_H
#define POLY_COMMON_H

#include "Logger.h"

#define ERROR(fmt, ...) \
            do { fprintf(stderr, "Error: "); fprintf(stderr, fmt, __VA_ARGS__); fprintf(stderr, "\n"); exit(1); } while (0)

#define WARNING(fmt, ...) \
            do { fprintf(stderr, "Warning: "); fprintf(stderr, fmt, __VA_ARGS__); fprintf(stderr, "\n"); } while (0)


extern poly::Logger Log;

std::string GetCurrentWorkingDirectory();

extern std::string WindowsPathSeparator;
extern std::string UnixPathSeparator;

#endif //POLY_COMMON_H
