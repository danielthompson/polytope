//
// Created by Daniel Thompson on 3/5/18.
//

#include "PathTraceIntegrator.h"
#include "../structures/Sample.h"

namespace poly {

   Sample PathTraceIntegrator::GetSample(Ray &ray, int depth, int x, int y) {
      Ray current_ray = ray;
      unsigned num_bounces = 0;
      
      ReflectanceSpectrum src(1.f, 1.f, 1.f);
      SpectralPowerDistribution direct_spd;
      float back_pdf = 1;
      
      bool debug = false;
#ifndef NDEBUG
      if (x == 665 && y == 347) {
         debug = false;
         
//         printf("ro: %f %f %f\n", ray.Origin.x, ray.Origin.y, ray.Origin.z);
//         printf("rd: %f %f %f\n", ray.Direction.x, ray.Direction.y, ray.Direction.z);
      }
#endif
      while (true) {
         if (src.is_zero())
            return Sample(SpectralPowerDistribution());
//         current_ray.x = x;
//         current_ray.y = y;
//         current_ray.bounce = num_bounces;
         Intersection intersection = Scene->GetNearestShape(current_ray, x, y);
         
         SpectralPowerDistribution bb_spd;
         
         bb_spd.r = 255.f - (float)(intersection.num_bb_hits) * 2.f;
         bb_spd.g = 255.f - (float)(intersection.num_bb_hits)* 2.f;
         bb_spd.b = 255.f - (float)(intersection.num_bb_hits)* 2.f;
         
         return Sample(bb_spd);
         
         if (debug) {
            printf("bounce %i: \n", num_bounces);
            //printf("t: %f\n", ray.MinT);
         }
         
         if (!intersection.Hits) {
            SpectralPowerDistribution spd;
            if (Scene->Skybox != nullptr) {
               spd = Scene->Skybox->GetSpd(ray.Direction) * src;
            }
            
            spd += direct_spd;
            return Sample(spd);
         }

         if (debug) {
            printf("hit face index %i\n", intersection.faceIndex);
         }
         
         if (intersection.Shape->is_light()) {
            return Sample((*(intersection.Shape->spd) + direct_spd) * src);
         }

         // base case
         if (num_bounces >= MaxDepth) {
            if (debug) {
               printf("max depth\n");
            }
            return Sample(SpectralPowerDistribution());
         } else {
            num_bounces++;

            float current_pdf;
            
            ReflectanceSpectrum refl;
            
            const poly::Vector local_incoming = intersection.WorldToLocal(current_ray.Direction);
            const poly::Vector local_outgoing = intersection.Shape->Material->BRDF->sample(local_incoming, refl,
                                                                                               current_pdf);
            const poly::Vector world_outgoing = intersection.LocalToWorld(local_outgoing);


            current_ray = Ray(intersection.Location, world_outgoing);
            current_ray.OffsetOrigin(intersection.bent_normal, poly::OffsetEpsilon);
            if (debug) {
               printf("n: %f %f %f\n", intersection.geo_normal.x, intersection.geo_normal.y, intersection.geo_normal.z);
               printf("o: %f %f %f\n", current_ray.Origin.x, current_ray.Origin.y, current_ray.Origin.z);
               printf("d: %f %f %f\n", current_ray.Direction.x, current_ray.Direction.y, current_ray.Direction.z);

            }
            src = src * refl; //intersection.Shape->Material->ReflectanceSpectrum;

            // add direct light 
            // 0. TODO if brdf is delta, continue 
            // 1. choose a light source
            
//            for (const auto light : Scene->Lights) {
//               // 2. get random point on light
//               Point light_point = light->random_surface_point();
//               
//               // 3. calculate reflected spd given BRDF for (intersection - light_point) -> -incoming
//               Vector light_to_intersection_local = intersection.Location - light_point;
//               Vector light_to_intersection_world = intersection.WorldToLocal(light_to_intersection_local);
//               
//               // 4. calculate BRDF for light_to_intersection -> incoming
//               ReflectanceSpectrum light_refl;
//               float light_pdf = 0.0f;
//               // TODO should use brdf->f()
//               intersection.Shape->Material->BRDF->sample(light_to_intersection_local, light_refl,light_pdf);
//               
//               // 5. if brdf == 0, continue
//               if (light_refl.is_black())
//                  continue;
//               
//               // 6. if light -> intersection is occluded, continue
//               Ray light_ray(current_ray.Origin, -light_to_intersection_world);
//               Intersection light_intersection = Scene->GetNearestShape(light_ray, x, y);
//               if (light_intersection.Hits && light_intersection.Shape != light)
//                  continue;
//               
//               direct_spd = direct_spd + ((*light->spd) * (src * light_refl));
//            }
            

            
            // direct_spd = direct_spd + src * [2] * light's spd
            
            back_pdf *= current_pdf;
         }
      }
   }
}