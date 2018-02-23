//
// Created by Daniel Thompson on 2/21/18.
//

#include "PixelRunner.h"

namespace Polytope {

   void Polytope::PixelRunner::Run() {
      for (int x = 0; x < _x; x++) {
         for (int y = 0; y < _y; y++) {

            Trace(x, y);
         }
      }

   }


}