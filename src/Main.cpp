//
// Created by Daniel Thompson on 2/18/18.
//

#include "Tracer.h"
#include "utilities/OptionsParser.h"
#include "utilities/Common.h"
#include <sstream>

#ifdef __CYGWIN__
#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#define BUFSIZE MAX_PATH
#endif

Polytope::Logger Log;

int main(int argc, char* argv[]) {

   try {
      Log = Polytope::Logger();

      Log.WithTime("Polytope started.");

      Polytope::Options options = Polytope::Options();

      if (argc > 0) {
         Polytope::OptionsParser parser(argc, argv);
         options = parser.Parse();
      }

      Polytope::Tracer tracer = Polytope::Tracer(options);
      tracer.Run();

      Log.WithTime("Exiting Polytope.");
   }
   catch(const std::exception&) {
      return EXIT_FAILURE;
   }
}

std::string GetCurrentWorkingDirectory() {
   TCHAR buffer[BUFSIZE];
   DWORD returnCode;
   returnCode = GetCurrentDirectory(BUFSIZE, buffer);
   if (returnCode == 0) {
      std::ostringstream oss;
      auto error = GetLastError();
      oss << "GetCurrentDirectory error (code " << error << ")";
      Log.WithTime(oss.str());
      return "";
   }
   if (returnCode > BUFSIZE) {
      std::ostringstream oss;
      oss << "Buffer too small (" << BUFSIZE << "); need " << returnCode << " characters...";
      Log.WithTime(oss.str());
      return "";
   }

   std::string temp(&buffer[0], &buffer[returnCode]);
   return temp;
}