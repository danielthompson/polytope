//
// Created by Daniel Thompson on 2/18/18.
//

#include "Tracer.h"
#include "utilities/OptionsParser.h"

int main(int argc, char* argv[]) {

   try {
      Polytope::Logger logger = Polytope::Logger();

      logger.LogTime("Polytope started.");

      Polytope::Options options = Polytope::Options();

      if (argc > 0) {
         Polytope::OptionsParser parser(argc, argv, logger);
         options = parser.Parse();
      }

      Polytope::Tracer tracer = Polytope::Tracer(logger, options);
      tracer.Run();

      logger.LogTime("Exiting Polytope.");
   }
   catch(const std::exception&) {
      return EXIT_FAILURE;
   }
}
