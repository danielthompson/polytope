//
// Created by Daniel Thompson on 3/5/18.
//

#include "PathTraceIntegrator.h"

namespace Polytope {

   Sample PathTraceIntegrator::GetSample(Ray &ray, int depth, int x, int y) {

      bool debug = false;
      if (x == 128 && y == 128)
         debug = true;
      
      Ray current_ray = ray;
      unsigned num_bounces = 0;
      
      ReflectanceSpectrum src(1.f, 1.f, 1.f);
      SpectralPowerDistribution direct_spd;
      float back_pdf = 1;
      
      while (true) {
         Intersection intersection = Scene->GetNearestShape(current_ray, x, y);
         
         if (!intersection.Hits) {
            SpectralPowerDistribution spd;
            if (Scene->Skybox != nullptr) {
               spd = Scene->Skybox->GetSpd(ray.Direction) * src;
            }
            
            spd += direct_spd;
            return Sample(spd);
         }

         if (intersection.Shape->is_light()) {
            return Sample((*(intersection.Shape->spd) + direct_spd) * src);
         }

         // base case
         if (num_bounces >= MaxDepth) {
            return Sample(SpectralPowerDistribution());
         } else {
            num_bounces++;

            float current_pdf;
            
            ReflectanceSpectrum refl;
            
            const Polytope::Vector local_incoming = intersection.WorldToLocal(current_ray.Direction);
            const Polytope::Vector local_outgoing = intersection.Shape->Material->BRDF->sample(local_incoming, refl,
                                                                                               current_pdf);
            const Polytope::Vector world_outgoing = intersection.LocalToWorld(local_outgoing);
            current_ray = Ray(intersection.Location, world_outgoing);
            current_ray.OffsetOrigin(intersection.Normal, Polytope::OffsetEpsilon);
            src = src * refl; //intersection.Shape->Material->ReflectanceSpectrum;

            // add direct light 
            // 0. TODO if brdf is delta, continue 
            // 1. choose a light source
            
            for (const auto light : Scene->Lights) {
               // 2. get random point on light
               Point light_point = light->random_surface_point();
               
               // 3. calculate reflected spd given BRDF for (intersection - light_point) -> -incoming
               Vector light_to_intersection_local = intersection.Location - light_point;
               Vector light_to_intersection_world = intersection.WorldToLocal(light_to_intersection_local);
               
               // 4. calculate BRDF for light_to_intersection -> incoming
               ReflectanceSpectrum light_refl;
               float light_pdf = 0.0f;
               // TODO should use brdf->f()
               intersection.Shape->Material->BRDF->sample(light_to_intersection_local, light_refl,light_pdf);
               
               // 5. if brdf == 0, continue
               if (light_refl.is_black())
                  continue;
               
               // 6. if light -> intersection is occluded, continue
               Ray light_ray(current_ray.Origin, -light_to_intersection_world);
               Intersection light_intersection = Scene->GetNearestShape(light_ray, x, y);
               if (light_intersection.Hits && light_intersection.Shape != light)
                  continue;
               
               direct_spd = direct_spd + ((*light->spd) * (src * light_refl));
            }
            

            
            // direct_spd = direct_spd + src * [2] * light's spd
            
            back_pdf *= current_pdf;
         }
      }
   }
}