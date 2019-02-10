//
// Created by Daniel on 29-Mar-18.
//

#ifndef POLYTOPE_OPTIONS_H
#define POLYTOPE_OPTIONS_H

namespace Polytope {

   class Options {
   public:
      unsigned int threads = 0;
      unsigned int samples = 4;
      std::string input_filename = "";
      std::string output_filename = "test.png";
      bool valid = true;
   };

}


#endif //POLYTOPE_OPTIONS_H
