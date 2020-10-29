//
// Created by Daniel on 21-Mar-18.
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdarg.h>

#include "Logger.h"

namespace poly {

   std::mutex log_mutex;

   std::string time_in_HH_MM_SS_MMM() {
      using namespace std::chrono;

      // get current time
      const auto now = system_clock::now();

      // get number of milliseconds for the current second
      // (remainder after division into seconds)
      const auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

      // convert to std::time_t in order to convert to std::tm (broken time)
      const auto timer = system_clock::to_time_t(now);

      // convert to broken time
      const std::tm bt = *std::localtime(&timer);

      std::ostringstream oss;

      oss << std::put_time(&bt, "%T"); // HH:MM:SS
      oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

      return oss.str();
   }

   void logger_reset_color() {
      printf("\033[0m");
   }

   void logger_set_yellow() {
         printf("\033[0;33m");
   }

   void logger_set_red() {
      printf("\033[1;31m");
   }

   void logger_set_blue() {
      printf("\033[0;34m");
   }
   
   void logger_set_green() {
      printf("\033[0;33m");
   }
   
   void Logger::log(log_level level, const char *format, va_list args) {
      const auto time = std::time(nullptr);
      const std::lock_guard<std::mutex> lock(log_mutex);
      if (level == log_level::LOG_ERROR) {
         std::cerr << "[" << std::put_time(std::localtime(&time), "%F ") << time_in_HH_MM_SS_MMM() << "] ";
      }
      else {
         std::cout << "[" << std::put_time(std::localtime(&time), "%F ") << time_in_HH_MM_SS_MMM() << "] ";
      }
      switch (level) {
         case log_level::LOG_DEBUG:
            logger_set_yellow();
            std::cout << "[D] ";
            break;
//         case log_level::LOG_INFO:
//            std::cout << "[I] ";
//            break;
         case log_level::LOG_WARNING:
            logger_set_red();
            std::cout << "[W] ";
            break;
         case log_level::LOG_ERROR:
            std::cerr << "[E] ";
            break;
      }
      if (level == log_level::LOG_ERROR) {
         vfprintf(stderr, format, args);
         fprintf(stderr, "\n");
      }
      else {
         vfprintf(stdout, format, args);
         printf("\n");
      }
      logger_reset_color();
   }

   void Logger::log(log_level level, const std::string& text) {
      va_list list;
      log(level, text.c_str(), list);
   }

   void Logger::debug(const char* format, ...) {
      if (LOG_LEVEL <= LOG_LEVEL_DEBUG) {
         va_list list;
         va_start(list, format);
         log(log_level::LOG_DEBUG, format, list);
         va_end(list);
      }
   }

   void Logger::info(const char* format, ...) {
      if (LOG_LEVEL <= LOG_LEVEL_INFO) {
         va_list list;
         va_start(list, format);
         log(log_level::LOG_INFO, format, list);
         va_end(list);
      }
   }

   void Logger::warning(const char* format, ...) {
      if (LOG_LEVEL <= LOG_LEVEL_WARNING) {
         va_list list;
         va_start(list, format);
         log(log_level::LOG_WARNING, format, list);
         va_end(list);
      }
   }

   void Logger::error(const char* format, ...) {
      if (LOG_LEVEL <= LOG_LEVEL_ERROR) {
         va_list list;
         va_start(list, format);
         log(log_level::LOG_ERROR, format, list);
         va_end(list);
      }
   }

   void Logger::debug(const std::string& text) {
      if (LOG_LEVEL <= LOG_LEVEL_DEBUG) {
         log(log_level::LOG_DEBUG, text);
      }
   }
   
   void Logger::info(const std::string& text) {
      if (LOG_LEVEL <= LOG_LEVEL_INFO) {
         log(log_level::LOG_INFO, text);
      }
   }
   
   void Logger::warning(const std::string& text) {
      if (LOG_LEVEL <= LOG_LEVEL_WARNING) {
         log(log_level::LOG_WARNING, text);
      }
   }
   
   void Logger::error(const std::string& text) {
      if (LOG_LEVEL <= LOG_LEVEL_ERROR) {
         log(log_level::LOG_ERROR, text);
      }
   }
}