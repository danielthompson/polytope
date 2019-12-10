//
// Created by Daniel Thompson on 12/10/19.
//

#ifndef POLYTOPE_GLOBAL_DEFINES_H
#define POLYTOPE_GLOBAL_DEFINES_H

#include <string>

#ifdef __CYGWIN__
   #include <windows.h>
   #include <stdio.h>
   #include <tchar.h>
   #define BUFSIZE MAX_PATH
#else
   #include <unistd.h>
#endif

std::string GetCurrentWorkingDirectory() {
#ifdef __CYGWIN__
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
#else
   char buffer[1000]; // hack
   char *answer = getcwd(buffer, sizeof(buffer));
   std::string s_cwd;
   if (answer) {
      s_cwd = answer;
   }

   return s_cwd;
#endif
}

#endif // POLYTOPE_GLOBAL_DEFINES_H