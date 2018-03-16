//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLYTOPE_ABSTRACTFILTER_H
#define POLYTOPE_ABSTRACTFILTER_H

#include "../structures/Sample.h"

namespace Polytope {

   /**
    * The purpose of a filter is to hold all samples and then render a given pixel
    * from those samples.
    */
   class AbstractFilter {
   public:

      // constructors
      AbstractFilter() = default;

      // methods
      virtual void AddSample(const Sample &sample) = 0;

      // destructors
      virtual ~AbstractFilter() { }

   };

}


#endif //POLYTOPE_ABSTRACTFILTER_H
