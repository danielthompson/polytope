//
// Created by Daniel Thompson on 3/5/18.
//

#include "PathTraceIntegrator.h"

namespace poly {

   poly::Sample PathTraceIntegrator::get_sample(poly::ray &ray, int depth, int x, int y) {
      poly::ray current_ray = ray;
      unsigned num_bounces = 0;

      poly::Sample sample;

      poly::ReflectanceSpectrum src(1.f, 1.f, 1.f);
      poly::SpectralPowerDistribution direct_spd;
      float back_pdf = 1;
      
      while (true) {
         if (src.is_zero())
            return sample;
         poly::intersection intersection = Scene->intersect(current_ray, x, y);

#ifdef POLYTOPEGL
         sample.intersections.push_back(intersection);
#endif
         
         if (!intersection.Hits) {
            poly::SpectralPowerDistribution spd;
            if (Scene->Skybox != nullptr) {
               spd = Scene->Skybox->GetSpd(ray.direction) * src;
            }
            
            spd += direct_spd;
            sample.SpectralPowerDistribution = spd;
            return sample;
         }
         
         if (intersection.shape->is_light()) {
            sample.SpectralPowerDistribution = (*(intersection.shape->spd) + direct_spd) * src;
            return sample;
         }

         // base case
         if (num_bounces >= MaxDepth) {
            return sample;
         } else {
            num_bounces++;

            float current_pdf;

            poly::ReflectanceSpectrum refl;
            
            const poly::vector local_incoming = intersection.world_to_local(current_ray.direction);
            const poly::vector local_outgoing = intersection.shape->material->BRDF->sample(local_incoming, intersection.u_tex_lerp, intersection.v_tex_lerp, refl,
                                                                                           current_pdf);
            const poly::vector world_outgoing = intersection.local_to_world(local_outgoing);
            
            LOG_DEBUG("bent normal dot outgoing: " << world_outgoing.dot(intersection.bent_normal));

            current_ray = poly::ray(intersection.location, world_outgoing);

#ifdef POLYTOPEGL
            sample.intersections.back().outgoing = world_outgoing;
#endif
            
            src = src * refl;
            back_pdf *= current_pdf;
         }
      }
   }
}