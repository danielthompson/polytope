//
// Created by daniel on 5/23/21.
//

#ifndef POLYTOPE_MESH_VAO_H
#define POLYTOPE_MESH_VAO_H

#include "../cpu/shading/brdf/lambert_brdf.h"
#include "../cpu/shading/brdf/mirror_brdf.h"
#include "../cpu/shapes/mesh.h"
#include "shaders.h"

namespace poly {
   class mesh_vao {
   private:
      unsigned vao_handle;
      std::shared_ptr<poly::shader_program> program;

      const std::string uniform_color = "colorIn";
      const std::string uniform_mvp = "mvp";

      std::vector<unsigned int> shapeIndexVector;
      std::vector<float> shapeVertexVector;
      std::vector<float> shapeNormalVector;
      
      glm::vec4 color;
      
   public:
      
      void init(const poly::Mesh* mesh) {

         program = poly::shader_cache::get_program({
             {
                   {"../src/gl/shape/vert.glsl", poly::shader_type::vert},
                   {"../src/gl/shape/frag.glsl", poly::shader_type::frag}
             },
             {
                   uniform_mvp,
                   uniform_color
             }
         });
         
         const unsigned int indices = mesh->mesh_geometry->num_faces * 3;

         shapeIndexVector = std::vector<unsigned int>(indices, 0);
         shapeVertexVector = std::vector<float>(indices * 3, 0.f);
         shapeNormalVector = std::vector<float>(indices * 3, 0.f);

         for (unsigned int i = 0; i < mesh->mesh_geometry->num_faces; i++) {

            // fix this garbage
            auto brdf = dynamic_cast<poly::LambertBRDF*>(mesh->material->BRDF.get());
            if (brdf != nullptr) {
               color = {brdf->refl.r, brdf->refl.g, brdf->refl.b, 1.0f};
            }
            else {
               auto brdf = dynamic_cast<poly::MirrorBRDF*>(mesh->material->BRDF.get());
               if (brdf != nullptr) {
                  color = {brdf->refl.r, brdf->refl.g, brdf->refl.b, 1.0f};
               }
            }
            
            std::shared_ptr<poly::mesh_geometry> geometry = mesh->mesh_geometry;

            poly::point p = {geometry->x_packed[geometry->fv0[i]],
                             geometry->y_packed[geometry->fv0[i]],
                             geometry->z_packed[geometry->fv0[i]]
            };
            mesh->object_to_world->apply_in_place(p);
            shapeVertexVector[9 * i] = p.x;
            shapeVertexVector[9 * i + 1] = p.y;
            shapeVertexVector[9 * i + 2] = p.z;

            p = {geometry->x_packed[geometry->fv1[i]],
                 geometry->y_packed[geometry->fv1[i]],
                 geometry->z_packed[geometry->fv1[i]]
            };
            mesh->object_to_world->apply_in_place(p);
            shapeVertexVector[9 * i + 3] = p.x;
            shapeVertexVector[9 * i + 4] = p.y;
            shapeVertexVector[9 * i + 5] = p.z;

            p = {geometry->x_packed[geometry->fv2[i]],
                 geometry->y_packed[geometry->fv2[i]],
                 geometry->z_packed[geometry->fv2[i]]
            };
            mesh->object_to_world->apply_in_place(p);
            shapeVertexVector[9 * i + 6] = p.x;
            shapeVertexVector[9 * i + 7] = p.y;
            shapeVertexVector[9 * i + 8] = p.z;

//         poly::Normal n = { geometry->nx_packed[geometry->fv0[i]],
//                            geometry->ny_packed[geometry->fv0[i]],
//                            geometry->nz_packed[geometry->fv0[i]]
//         };
            poly::normal n = {0, 0, 0};

            mesh->object_to_world->apply_in_place(n);
            shapeNormalVector[9 * i] = n.x;
            shapeNormalVector[9 * i + 1] = n.y;
            shapeNormalVector[9 * i + 2] = n.z;

//         n = {geometry->nx_packed[geometry->fv1[i]],
//              geometry->ny_packed[geometry->fv1[i]],
//              geometry->nz_packed[geometry->fv1[i]]
//         };
            n = {0, 0, 0};
            mesh->object_to_world->apply_in_place(n);
            shapeNormalVector[9 * i + 3] = n.x;
            shapeNormalVector[9 * i + 4] = n.y;
            shapeNormalVector[9 * i + 5] = n.z;

//         n = { geometry->nx_packed[geometry->fv2[i]],
//               geometry->ny_packed[geometry->fv2[i]],
//               geometry->nz_packed[geometry->fv2[i]]
//         };
            n = {0, 0, 0};
            mesh->object_to_world->apply_in_place(n);
            shapeNormalVector[9 * i + 6] = n.x;
            shapeNormalVector[9 * i + 7] = n.y;
            shapeNormalVector[9 * i + 8] = n.z;

            shapeIndexVector[3 * i] = 3 * i;
            shapeIndexVector[3 * i + 1] = 3 * i + 1;
            shapeIndexVector[3 * i + 2] = 3 * i + 2;
         }

         gl::glGenVertexArrays(1, &vao_handle);
         gl::glBindVertexArray(vao_handle);

         // buffer for vertex indices
         gl::GLuint shapeIndexBuffer;
         gl::glGenBuffers(1, &shapeIndexBuffer);
         gl::glBindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, shapeIndexBuffer);
         gl::glBufferData(gl::GL_ELEMENT_ARRAY_BUFFER, indices * sizeof(indices), &shapeIndexVector[0],
                          gl::GL_STATIC_DRAW);

         // buffer for vertex locations
         gl::GLuint shapeVertexBuffer;
         gl::glGenBuffers(1, &shapeVertexBuffer);
         gl::glBindBuffer(gl::GL_ARRAY_BUFFER, shapeVertexBuffer);
         gl::glBufferData(gl::GL_ARRAY_BUFFER, indices * 3 * sizeof(indices), &shapeVertexVector[0],
                          gl::GL_STATIC_DRAW);
         gl::glVertexAttribPointer(
               0,        // attribute 0 - must match layout in shader
               3,        // size
               gl::GL_FLOAT, // type
               gl::GL_FALSE, // normalized?
               0,  // stride
               (void *) 0 // array buffer offset
         );
         gl::glEnableVertexAttribArray(0);

         gl::GLuint shapeNormalBuffer;
         gl::glGenBuffers(1, &shapeNormalBuffer);
         gl::glBindBuffer(gl::GL_ARRAY_BUFFER, shapeNormalBuffer);
         gl::glBufferData(gl::GL_ARRAY_BUFFER, indices * 3 * sizeof(indices), &shapeNormalVector[0],
                          gl::GL_STATIC_DRAW);

         gl::glBindVertexArray(0);
      }

      void draw(const glm::mat4 &mvp) {
         glDisable(gl::GL_DEPTH_TEST);
         glEnable(gl::GL_BLEND);
         
         program->use();
         program->set_uniform(uniform_mvp, mvp);
         gl::glBindVertexArray(vao_handle);

         gl::glEnable(gl::GL_DEPTH_TEST);
         gl::glDisable(gl::GL_BLEND);

         program->set_uniform(uniform_color, color);
         gl::glPolygonMode(gl::GL_FRONT_AND_BACK, gl::GL_FILL);
         gl::glDrawElements(gl::GL_TRIANGLES, shapeIndexVector.size(), gl::GL_UNSIGNED_INT, (void*)0);

         gl::glDisable(gl::GL_DEPTH_TEST);
         gl::glEnable(gl::GL_BLEND);

         program->set_uniform(uniform_color, {1.0f, 1.0f, 1.0f, 0.16250f});
         gl::glPolygonMode(gl::GL_FRONT_AND_BACK, gl::GL_LINE);
         gl::glDrawElements(gl::GL_TRIANGLES, shapeIndexVector.size(), gl::GL_UNSIGNED_INT, (void*)0);
      }
   };
}

#endif //POLYTOPE_MESH_VAO_H
