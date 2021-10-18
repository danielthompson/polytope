//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_SAMPLERS_H
#define POLY_SAMPLERS_H

#include "../../common/structures/point2.h"

namespace poly {

   class abstract_sampler {
   public:

      // methods
      virtual point2f GetSample(int x, int y) const = 0;

      virtual void GetSamples(point2f points[], unsigned int number, int x, int y) const = 0;

      // destructors

      virtual ~abstract_sampler() = default;

   };

   class CenterSampler : public abstract_sampler {
   public:
      point2f GetSample(int x, int y) const override;

      void GetSamples(point2f points[], unsigned int number, int x, int y) const override;

      ~CenterSampler() override = default;
   };

   class HaltonSampler : public abstract_sampler {
   public:
      point2f GetSample(int x, int y) const override;

      void GetSamples(point2f points[], unsigned int number, int x, int y) const override;

   };

   class GridSampler : public abstract_sampler {
   public:
      point2f GetSample(int x, int y) const override;

      ~GridSampler() override = default;

      void GetSamples(point2f points[], unsigned int number, int x, int y) const override;
   };
   
   class random_sampler : public abstract_sampler {
   public:
      point2f GetSample(int x, int y) const override;

      ~random_sampler() override = default;

      void GetSamples(point2f points[], unsigned int number, int x, int y) const override;
   };
   
}


#endif //POLY_SAMPLERS_H
