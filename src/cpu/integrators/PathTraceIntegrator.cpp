//
// Created by Daniel Thompson on 3/5/18.
//

#include "PathTraceIntegrator.h"

namespace poly {

   poly::Sample PathTraceIntegrator::get_sample(poly::ray &ray, int depth, int x, int y) {
      Log.debug("PathTraceIntegrator::this: %p", this);
      poly::ray current_ray = ray;
      unsigned num_bounces = 0;

      poly::Sample sample;

      poly::ReflectanceSpectrum src(1.f, 1.f, 1.f);
      poly::SpectralPowerDistribution direct_spd;
      float back_pdf = 1;
      
      bool debug = false;
      if (x == 50 && y == 200) {
         debug = true;
      }
      
      if (debug) {
         printf("o: %f %f %f\n", current_ray.origin.x, current_ray.origin.y, current_ray.origin.z);
         printf("d: %f %f %f\n\n", current_ray.direction.x, current_ray.direction.y, current_ray.direction.z);
      }
#ifndef NDEBUG

#endif
      while (true) {
         if (src.is_zero())
            return sample;
//         current_ray.x = x;
//         current_ray.y = y;
//         current_ray.bounce = num_bounces;
         poly::intersection intersection = Scene->intersect(current_ray, x, y);

#ifdef POLYTOPEGL
         sample.intersections.push_back(intersection);
#endif
         
//         SpectralPowerDistribution bb_spd;
//         
//         bb_spd.r = 255.f - (float)(intersection.num_bb_hits) * 2.f;
//         bb_spd.g = 255.f - (float)(intersection.num_bb_hits)* 2.f;
//         bb_spd.b = 255.f - (float)(intersection.num_bb_hits)* 2.f;
//         
//         return Sample(bb_spd);
         
         if (debug) {
            printf("bounce %i: \n", num_bounces);
            //printf("t: %f\n", ray.MinT);
         }
         
         if (!intersection.Hits) {
            poly::SpectralPowerDistribution spd;
            if (Scene->Skybox != nullptr) {
               spd = Scene->Skybox->GetSpd(ray.direction) * src;
            }
            
            spd += direct_spd;
            sample.SpectralPowerDistribution = spd;
            return sample;
         }

         if (debug) {
            printf("hit mesh index %i face index %i\n", intersection.mesh_index, intersection.face_index);
         }
         
         if (intersection.shape->is_light()) {
            sample.SpectralPowerDistribution = (*(intersection.shape->spd) + direct_spd) * src;
            return sample;
         }

         // base case
         if (num_bounces >= MaxDepth) {
            if (debug) {
               printf("max depth\n");
            }
            return sample;
         } else {
            num_bounces++;

            float current_pdf;

            poly::ReflectanceSpectrum refl;
            
            const poly::vector local_incoming = intersection.world_to_local(current_ray.direction);
            const poly::vector local_outgoing = intersection.shape->material->BRDF->sample(local_incoming, intersection.u_tex_lerp, intersection.v_tex_lerp, refl,
                                                                                           current_pdf);
            const poly::vector world_outgoing = intersection.local_to_world(local_outgoing);

            bool whoops = false;
            if (x == 128 && y == 128 && std::isnan(world_outgoing.x))
               whoops = true;

            current_ray = poly::ray(intersection.location, world_outgoing);
            
            //current_ray.OffsetOrigin(intersection.bent_normal, poly::OffsetEpsilon);
            if (debug) {
               printf("gn: %f %f %f\n", intersection.geo_normal.x, intersection.geo_normal.y, intersection.geo_normal.z);
               printf("uvw: %f %f %f\n", intersection.u, intersection.v, intersection.w);
               printf("o: %f %f %f\n", current_ray.origin.x, current_ray.origin.y, current_ray.origin.z);
               printf("d: %f %f %f\n", current_ray.direction.x, current_ray.direction.y, current_ray.direction.z);

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
//               Intersection light_intersection = Scene->intersect(light_ray, x, y);
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