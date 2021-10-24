//
// Created by Daniel Thompson on 2/21/18.
//

#ifndef POLY_SAMPLERS_H
#define POLY_SAMPLERS_H

#include "../../common/structures/point2.h"
#include "random_number_generator.h"

namespace poly {

   class abstract_sampler {
   public:
      virtual void get_samples(poly::point2f points[], unsigned int number, int x, int y) = 0;
      virtual ~abstract_sampler() = default;
   };

   class center_sampler : public abstract_sampler {
   public:
      void get_samples(poly::point2f points[], unsigned int number, int x, int y) override;
   };

   class halton_sampler : public abstract_sampler {
   public:
      void get_samples(poly::point2f points[], unsigned int number, int x, int y) override;
   };

   class grid_sampler : public abstract_sampler {
   public:
      void get_samples(poly::point2f points[], unsigned int number, int x, int y) override;
   };
   
   class random_sampler : public abstract_sampler {
   public:
      void get_samples(poly::point2f points[], unsigned int number, int x, int y) override;
   };
}


#endif //POLY_SAMPLERS_H
