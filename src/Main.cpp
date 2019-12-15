//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <sstream>
#include "Tracer.h"
#include "utilities/OptionsParser.h"
#include "utilities/Common.h"

#ifdef __CYGWIN__
#include "platforms/win32-cygwin.h"
#endif

Polytope::Logger Log;

void segfaultHandler(int signalNumber) {
   Log.WithTime("Detected a segfault. Stacktrace to be implemented...");
#ifdef __CYGWIN__
   //printStack();
#endif
   exit(signalNumber);
}

void signalHandler(int signalNumber) {
   std::ostringstream oss;
   oss << "Received interrupt signal " << signalNumber << ", aborting.";
   Log.WithTime(oss.str());
   exit(signalNumber);
}

bool hasAbortedOnce = false;

void userAbortHandler(int signalNumber) {
   if (hasAbortedOnce) {
      Log.WithTime("Aborting at user request.");
      exit(signalNumber);
   }
   else {
      Log.WithTime("Detected Ctrl-C keypress. Ignoring since it's the first time. Press Ctrl-C again to really quit.");
      hasAbortedOnce = true;
   }
}

int main(int argc, char* argv[]) {

//   signal(SIGSEGV, segfaultHandler);
//
//   signal(SIGINT, userAbortHandler);
//
//   signal(SIGABRT, signalHandler);
//   signal(SIGFPE, signalHandler);
//   signal(SIGILL, signalHandler);
//   signal(SIGTERM, signalHandler);


   try {
      Log = Polytope::Logger();

      Polytope::Options options = Polytope::Options();

      if (argc > 0) {
         Polytope::OptionsParser parser(argc, argv);
         options = parser.Parse();
      }

      if (options.help) {
         std::cout << "Polytope by Daniel A. Thompson, built on " << __DATE__ << std::endl;
         fprintf(stderr, R"(
Usage: polytope [options] -inputfile <filename> [-outputfile <filename>]

Rendering options:
   -threads <n>      Number of CPU threads to use for rendering. Optional;
                     defaults to the number of detected logical cores.
   -samples <n>      Number of samples to use per pixel. Optional; overrides
                     the number of samples specified in the scene file.

File options:
   -inputfile        The scene file to render. Currently, PBRT is the only
                     supported file format. Optional but strongly encouraged;
                     defaults to a boring example scene.
   -outputfile       The filename to render to. Currently, PNG is the only
                     supported output file format. Optional; overrides the
                     output filename specified in the scene file, if any;
                     defaults to the input file name (with .png extension).

Other:
   --help            Print this help text and exit.)");
         std::cout << std::endl;
         exit(0);
      }


      Polytope::Tracer tracer = Polytope::Tracer(options);
      tracer.Run();

      Log.WithTime("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
