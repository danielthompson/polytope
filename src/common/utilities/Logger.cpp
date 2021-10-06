//
// Created by Daniel on 21-Mar-18.
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdarg.h>
#include <fstream>

#include "Logger.h"

namespace poly {
   
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

   void log_time(std::ostream& stream) {
      logger_reset_color();
      const auto time = std::time(nullptr);
      stream << "[" << std::put_time(std::localtime(&time), "%F ") << time_in_HH_MM_SS_MMM() << "] ";
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

   std::ostream &Logger::debug() {
      log_time(std::cout);
      logger_set_yellow();
      std::cout << "[D] ";
      return std::cout;
   }

   std::ostream &Logger::info() {
      log_time(std::cout);
      return std::cout;
   }

   std::ostream &Logger::error() {
      log_time(std::cout);
      logger_set_red();
      std::cout << "[E] ";
      return std::cout;
   }

   std::ostream &Logger::warning() {
      log_time(std::cout);
      logger_set_red();
      std::cout << "[W] ";
      return std::cout;
   }
}