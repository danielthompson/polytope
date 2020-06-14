//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_SAMPLERS_H
#define POLY_SAMPLERS_H

#include "../../common/structures/Point2.h"

namespace poly {

   class AbstractSampler {
   public:

      // methods
      virtual Point2f GetSample(int x, int y) const = 0;

      virtual void GetSamples(Point2f points[], unsigned int number, int x, int y) const = 0;

      // destructors

      virtual ~AbstractSampler() = default;

   };

   class CenterSampler : public AbstractSampler {
   public:
      Point2f GetSample(int x, int y) const override;

      void GetSamples(Point2f points[], unsigned int number, int x, int y) const override;

      ~CenterSampler() override = default;
   };

   class HaltonSampler : public AbstractSampler {
   public:
      Point2f GetSample(int x, int y) const override;

      void GetSamples(Point2f points[], unsigned int number, int x, int y) const override;

   };

   class GridSampler : public AbstractSampler {
   public:
      Point2f GetSample(int x, int y) const override;

      ~GridSampler() override = default;

      void GetSamples(Point2f points[], unsigned int number, int x, int y) const override;
   };
   
}


#endif //POLY_SAMPLERS_H
