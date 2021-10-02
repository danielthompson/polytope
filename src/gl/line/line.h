//
// Created by daniel on 9/30/21.
//

#ifndef POLYTOPE_QUAD_SPHERE_VAO_H
#define POLYTOPE_QUAD_SPHERE_VAO_H

#endif //POLYTOPE_QUAD_SPHERE_VAO_H

namespace poly {
   class line {
   private:

      // all nodes
      unsigned all_vao_handle;

      gl::GLuint all_vertex_buffer_handle;
      std::vector<float> all_vertex_buffer;

      std::shared_ptr<poly::shader_program> all_program;

      const std::string uniform_color = "colorIn";
      const std::string uniform_mvp = "mvp";
      glm::vec4 color;

   public:
      line(const poly::point& start, const poly::point& end, const glm::vec4& color) : color(color) {

         all_program = poly::shader_cache::get_program({
                                                             {
                                                                   {"../src/gl/line/vert.glsl", poly::shader_type::vert},
                                                                   {"../src/gl/line/frag.glsl", poly::shader_type::frag}
                                                             },
                                                             {
                                                                   uniform_mvp,
                                                                   uniform_color
                                                             }
                                                       });

         all_vertex_buffer = {
            start.x,
            start.y,
            start.z,
            end.x,
            end.y,
            end.z
         };
         

         // all nodes
         gl::glGenVertexArrays(1, &all_vao_handle);
         gl::glBindVertexArray(all_vao_handle);

         gl::glGenBuffers(1, &all_vertex_buffer_handle);
         gl::glBindBuffer(gl::GL_ARRAY_BUFFER, all_vertex_buffer_handle);
         gl::glBufferData(gl::GL_ARRAY_BUFFER, 
                          all_vertex_buffer.size() * sizeof(float), 
                          &all_vertex_buffer[0],
                          gl::GL_STATIC_DRAW);
         gl::glVertexAttribPointer(
               0,        // attribute 0 - must match layout in shader
               3,        // size
               gl::GL_FLOAT, // type
               gl::GL_FALSE, // normalized?
               3 * sizeof(float),  // stride
               (void *) 0 // array buffer offset
         );
         gl::glEnableVertexAttribArray(0);
         gl::glBindBuffer(gl::GL_ARRAY_BUFFER, 0);
         gl::glBindVertexArray(0);
      }

      void draw(const glm::mat4 &mvp) const {
         gl::glEnable(gl::GL_DEPTH_TEST);
         all_program->use();
         all_program->set_uniform(uniform_color, color);
         all_program->set_uniform(uniform_mvp, mvp);
         gl::glBindVertexArray(all_vao_handle);
         gl::glDrawArrays(gl::GL_LINES, 0, 2);
      }
      
      ~line() {
         // delete VBO and VAO
         gl::glDeleteVertexArrays(1, &all_vao_handle);
         gl::glDeleteBuffers(1, &all_vertex_buffer_handle);
      }
   };
}