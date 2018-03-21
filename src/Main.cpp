//
// Created by Daniel Thompson on 2/18/18.
//

#include "Tracer.h"


int main() {

   Polytope::Logger logger = Polytope::Logger();
   logger.log("Polytope started.");

   Polytope::Tracer tracer = Polytope::Tracer(logger);
   tracer.Run();

   logger.log("Exiting Polytope.");

}
