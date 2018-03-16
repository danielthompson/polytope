//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLYTOPE_BOXFILTER_H
#define POLYTOPE_BOXFILTER_H

#include "AbstractFilter.h"

namespace Polytope {

   class BoxFilter : public AbstractFilter {
   public:
      BoxFilter() : AbstractFilter() { }

      void AddSample(const Sample &sample) override;
   };

}


#endif //POLYTOPE_BOXFILTER_H
