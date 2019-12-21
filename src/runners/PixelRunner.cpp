//
// Created by Daniel Thompson on 2/21/18.
//

#include "PixelRunner.h"

namespace Polytope {

   void Polytope::PixelRunner::Run(int threadId) {
      for (int y = 0; y < Bounds.y; y++) {
         for (int x = 0; x < Bounds.x; x++) {
            Trace(x, y);
         }
      }
   }
}