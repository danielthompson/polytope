//
// Created by Daniel Thompson on 2/21/18.
//

#include <iostream>
#include "AbstractRunner.h"
#include "../structures/Ray.h"

namespace poly {

   void AbstractRunner::Trace(const int x, const int y) const {

      Point2f points[NumSamples];

      Sampler->GetSamples(points, NumSamples, x, y);
      
      for (unsigned int i = 0; i < NumSamples; i++) {

         Point2f sampleLocation = points[i];

         bool debug = false;
         
         
         Ray ray = Scene->Camera->get_ray_for_pixel(sampleLocation);
         if (x == 200 && y == 200)
            debug = true;
         
         Sample sample = Integrator->GetSample(ray, 0, x, y);
         Film->AddSample(sampleLocation, sample);
      }
   }

   void AbstractRunner::Output() const {
      Film->Output();
   };
}