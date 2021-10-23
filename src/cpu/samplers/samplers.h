//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_SAMPLERS_H
#define POLY_SAMPLERS_H

#include "../../common/structures/point2.h"

namespace poly {

   class abstract_sampler {
   public:
      virtual void get_samples(poly::point2f points[], unsigned int number, int x, int y) const = 0;
      virtual ~abstract_sampler() = default;
   };

   class center_sampler : public abstract_sampler {
   public:
      void get_samples(poly::point2f points[], unsigned int number, int x, int y) const override;
   };

   class halton_sampler : public abstract_sampler {
   public:
      void get_samples(poly::point2f points[], unsigned int number, int x, int y) const override;
   };

   class grid_sampler : public abstract_sampler {
   public:
      void get_samples(poly::point2f points[], unsigned int number, int x, int y) const override;
   };
   
   class random_sampler : public abstract_sampler {
   public:
      random_sampler();
      void get_samples(poly::point2f points[], unsigned int number, int x, int y) const override;
   };
}


#endif //POLY_SAMPLERS_H
