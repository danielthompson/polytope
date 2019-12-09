//
// Created by Daniel on 21-Mar-18.
//

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#include "Logger.h"

namespace Polytope {

   std::string Logger::time_in_HH_MM_SS_MMM() const {
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

   void Logger::ErrorWithTime(const std::string &text) const {
      const auto time = std::time(nullptr);
      std::cerr
   }

   void Logger::WithTime(const std::string& text) const {
      const auto time = std::time(nullptr);
      std::cout << "[" << std::put_time(std::localtime(&time), "%F ") << time_in_HH_MM_SS_MMM() << "] "; // ISO 8601 format.
      std::cout << text << std::endl;
   }

   void Logger::WithTime(const std::ostream out, const std::string &text) const {
      const auto time = std::time(nullptr);
      std::cout << "[" << std::put_time(std::localtime(&time), "%F ") << time_in_HH_MM_SS_MMM() << "] "; // ISO 8601 format.
      std::cout << text << std::endl;
   }

   void Logger::logThread(const std::string& text) const {
      const auto time = std::time(nullptr);

      const std::thread::id threadID = std::this_thread::get_id();

      std::cout << "[" << std::put_time(std::localtime(&time), "%F ") << time_in_HH_MM_SS_MMM() << "] "; // ISO 8601 format.
      std::cout << "Thread [" << threadID << "] ";
      std::cout << text << std::endl;
   }

   void Logger::Log(const std::string& text) const {
      std::cout << text << std::endl;
   }

}