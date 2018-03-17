//
// Created by Daniel on 16-Mar-18.
//

#ifndef POLYTOPE_BOXFILTER_H
#define POLYTOPE_BOXFILTER_H

#include <vector>
#include "boost/multi_array.hpp"
#include "AbstractFilter.h"
#include "../structures/Point2.h"


namespace Polytope {

   class BoxFilter : public AbstractFilter {
   public:
      explicit BoxFilter(const Polytope::Bounds &bounds)
            : AbstractFilter(bounds), _data(array_type(boost::extents[bounds.x][bounds.y][1])) { }

      Sample Output(const Point2i &pixel) override;

      void AddSample(const Point2f &location, const Sample &sample) override;

   private:
      typedef boost::multi_array<Sample, 3> array_type;
      typedef array_type::index index;

      array_type _data;


   };

}


#endif //POLYTOPE_BOXFILTER_H
