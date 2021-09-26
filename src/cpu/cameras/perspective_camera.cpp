//
// Created by dthompson on 20 Feb 18.
//

#include <cmath>
#include <cassert>
#include <iostream>
#include "perspective_camera.h"
#include "../constants.h"
#include "../structures/Vectors.h"
#include "../structures/stats.h"

extern thread_local poly::stats thread_stats;

namespace poly {
   
   perspective_camera::perspective_camera(const poly::camera_settings &settings,
                                          const poly::transform &camera_to_world,
                                          const bool left_handed)
         : abstract_camera(settings, camera_to_world),
           left_handed(left_handed) {
      one_over_width = 1.0f / (float)settings.bounds.x;
      one_over_height = 1.0f / (float)settings.bounds.y;
      aspect_ratio = (float) settings.bounds.x * one_over_height;
      tan_fov_over_2 = (float)tan(settings.field_of_view * poly::PIOver180 * .5f);
   }

   poly::ray perspective_camera::get_ray_for_pixel(const poly::point2f pixel) {
      const float pixel_ndc_x = (pixel.x/* + 0.5f*/) * one_over_width;
      const float pixel_ndc_y = (pixel.y/* + 0.5f*/) * one_over_height;
      
      const float pixel_camera_x = (2 * pixel_ndc_x - 1) * aspect_ratio * tan_fov_over_2;
      const float pixel_camera_y = (1 - 2 * pixel_ndc_y) * tan_fov_over_2;
      
      const float z = left_handed ? 1 : -1;

      poly::point imagePlanePixelInCameraSpace = {pixel_camera_x, pixel_camera_y, z };

      poly::vector direction = {imagePlanePixelInCameraSpace.x, imagePlanePixelInCameraSpace.y, imagePlanePixelInCameraSpace.z };

      direction.normalize();

      const poly::ray camera_space_ray = {DefaultOrigin, direction };
      poly::ray world_space_ray = camera_to_world.apply(camera_space_ray);

      world_space_ray.direction.normalize();

      thread_stats.num_camera_rays++;
      return world_space_ray;
   }

   poly::ray perspective_camera::get_ray_for_ndc(const poly::point2f ndc) {
      assert(ndc.x >= 0);
      assert(ndc.x <= 1.f);
      assert(ndc.y >= 0);
      assert(ndc.y <= 1.f);
      const float pixel_ndc_x = ndc.x;
      const float pixel_ndc_y = ndc.y;

      const float pixel_camera_x = (2 * pixel_ndc_x - 1) * aspect_ratio * tan_fov_over_2;
      const float pixel_camera_y = (1 - 2 * pixel_ndc_y) * tan_fov_over_2;

      const float z = left_handed ? 1 : -1;

      poly::point image_plane_pixel_in_camera_space = {pixel_camera_x, pixel_camera_y, z };

      poly::vector direction = {image_plane_pixel_in_camera_space.x, 
                                image_plane_pixel_in_camera_space.y, 
                                image_plane_pixel_in_camera_space.z };

      direction.normalize();

      poly::ray camera_space_ray = {DefaultOrigin, direction };
      poly::ray world_space_ray = camera_to_world.apply(camera_space_ray);

      world_space_ray.direction.normalize();

      thread_stats.num_camera_rays++;
      return world_space_ray;
   }
}