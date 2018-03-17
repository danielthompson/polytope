//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLYTOPE_ABSTRACTFILTER_H
#define POLYTOPE_ABSTRACTFILTER_H

#include "../structures/Sample.h"
#include "../structures/Point2.h"

namespace Polytope {

   /**
    * The purpose of a filter is to hold all samples and then render a given pixel
    * from those samples.
    */
   class AbstractFilter {
   public:

      // constructors
      explicit AbstractFilter(const Bounds &bounds) : Bounds(bounds) { }

      // methods
      virtual void AddSample(const Point2f &location, const Sample &sample) = 0;

      virtual Sample Output(const Point2i &pixel) = 0;

      // destructors
      virtual ~AbstractFilter() { }

      const Bounds Bounds;
   };

}


#endif //POLYTOPE_ABSTRACTFILTER_H
