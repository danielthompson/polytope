//
// Created by Daniel on 29-Mar-18.
//

#ifndef POLY_OPTIONS_H
#define POLY_OPTIONS_H

namespace poly {

   class Options {
   public:
      unsigned int threads = 0;
      bool threadsSpecified = false;

      unsigned int samples = 16;
      bool samplesSpecified = false;

      std::string input_filename = "";
      bool inputSpecified = false;

      std::string output_filename = "polytope.png";
      bool outputSpecified = false;

      bool gl = false;

      bool valid = true;
      bool help = false;
   };
}

#endif //POLY_OPTIONS_H
