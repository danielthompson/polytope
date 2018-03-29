//
// Created by Daniel Thompson on 2/18/18.
//

#include "Tracer.h"
#include "utilities/OptionsParser.h"


int main(int argc, char* argv[]) {

   if (argc > 0) {
      Polytope::OptionsParser parser(argc, argv);
      Polytope::Options options = parser.Parse();

   }

   Polytope::Logger logger = Polytope::Logger();
   logger.log("Polytope started.");

   Polytope::Tracer tracer = Polytope::Tracer(logger);
   //tracer.Run();

   logger.log("Exiting Polytope.");

}
