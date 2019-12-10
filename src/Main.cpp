//
// Created by Daniel Thompson on 2/18/18.
//

#include "Tracer.h"
#include "utilities/OptionsParser.h"
#include "utilities/Common.h"

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
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}
