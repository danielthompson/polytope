//
// Created by daniel on 9/30/21.
//

#ifndef POLYTOPE_QUAD_SPHERE_VAO_H
#define POLYTOPE_QUAD_SPHERE_VAO_H

#endif //POLYTOPE_QUAD_SPHERE_VAO_H

namespace poly {
   class quad_sphere_vao {
   private:

      // all nodes
      unsigned all_vao_handle;

      gl::GLuint all_index_buffer_handle;
      std::vector<unsigned int> all_index_buffer;

      gl::GLuint all_vertex_buffer_handle;
      std::vector<float> all_vertex_buffer;

      std::vector<unsigned int> all_lines_index_buffer;

      std::shared_ptr<poly::shader_program> all_program;

      const std::string uniform_color = "colorIn";
      const std::string uniform_mvp = "mvp";

      // selected node
      unsigned selected_vao_handle;
      unsigned selected_node_index_handle;

      unsigned selected_vbo_handle;
      float selected_vertex_buffer[24];
      unsigned selected_index_buffer[24] {
            // x lines
            0, 1,
            3, 2,
            5, 6,
            4, 7,

            // y lines
            0, 3,
            1, 2,
            4, 5,
            7, 6,

            // z lines
            0, 4,
            1, 7,
            3, 5,
            2, 6,
      };

   public:
      void select_node(poly::bvh_node* node) {
         const point low = node->bb.p0;
         const point high = node->bb.p1;

         selected_vertex_buffer[0] = low.x;
         selected_vertex_buffer[1] = low.y;
         selected_vertex_buffer[2] = low.z;

         selected_vertex_buffer[3] = high.x;
         selected_vertex_buffer[4] = low.y;
         selected_vertex_buffer[5] = low.z;

         selected_vertex_buffer[6] = high.x;
         selected_vertex_buffer[7] = high.y;
         selected_vertex_buffer[8] = low.z;

         selected_vertex_buffer[9] = low.x;
         selected_vertex_buffer[10] = high.y;
         selected_vertex_buffer[11] = low.z;

         selected_vertex_buffer[12] = low.x;
         selected_vertex_buffer[13] = low.y;
         selected_vertex_buffer[14] = high.z;

         selected_vertex_buffer[15] = low.x;
         selected_vertex_buffer[16] = high.y;
         selected_vertex_buffer[17] = high.z;

         selected_vertex_buffer[18] = high.x;
         selected_vertex_buffer[19] = high.y;
         selected_vertex_buffer[20] = high.z;

         selected_vertex_buffer[21] = high.x;
         selected_vertex_buffer[22] = low.y;
         selected_vertex_buffer[23] = high.z;

         gl::glBindVertexArray(selected_vao_handle);
         gl::glBindBuffer(gl::GL_ARRAY_BUFFER, selected_vbo_handle);
         gl::glBufferData(gl::GL_ARRAY_BUFFER, 24 * sizeof(float), &selected_vertex_buffer[0], gl::GL_STATIC_DRAW);
         gl::glBindVertexArray(0);
      }

      void init(const poly::bvh& bvh) {

         all_program = poly::shader_cache::get_program({
                                                             {
                                                                   {"../src/gl/bbox/vert.glsl", poly::shader_type::vert},
                                                                   {"../src/gl/bbox/frag.glsl", poly::shader_type::frag}
                                                             },
                                                             {
                                                                   uniform_mvp,
                                                                   uniform_color
                                                             }
                                                       });

         std::queue<std::pair<poly::bvh_node *, unsigned int>> queue;

         if (bvh.root != nullptr) {
            queue.push(std::make_pair(bvh.root, 0));

            unsigned int index = 0;

            while (!queue.empty()) {
               const auto pair = queue.front();

               const poly::bvh_node *node = pair.first;
               const unsigned int nodeDepth = pair.second;
               queue.pop();

               const point low = node->bb.p0;
               const point high = node->bb.p1;

               // add BB box to shapes
               all_vertex_buffer.push_back(low.x);
               all_vertex_buffer.push_back(low.y);
               all_vertex_buffer.push_back(low.z);

               all_vertex_buffer.push_back(high.x);
               all_vertex_buffer.push_back(low.y);
               all_vertex_buffer.push_back(low.z);

               all_vertex_buffer.push_back(high.x);
               all_vertex_buffer.push_back(high.y);
               all_vertex_buffer.push_back(low.z);

               all_vertex_buffer.push_back(low.x);
               all_vertex_buffer.push_back(high.y);
               all_vertex_buffer.push_back(low.z);

               all_vertex_buffer.push_back(low.x);
               all_vertex_buffer.push_back(low.y);
               all_vertex_buffer.push_back(high.z);

               all_vertex_buffer.push_back(low.x);
               all_vertex_buffer.push_back(high.y);
               all_vertex_buffer.push_back(high.z);

               all_vertex_buffer.push_back(high.x);
               all_vertex_buffer.push_back(high.y);
               all_vertex_buffer.push_back(high.z);

               all_vertex_buffer.push_back(high.x);
               all_vertex_buffer.push_back(low.y);
               all_vertex_buffer.push_back(high.z);

               // x lines

               all_lines_index_buffer.push_back(index + 0);
               all_lines_index_buffer.push_back(index + 1);

               all_lines_index_buffer.push_back(index + 3);
               all_lines_index_buffer.push_back(index + 2);

               all_lines_index_buffer.push_back(index + 5);
               all_lines_index_buffer.push_back(index + 6);

               all_lines_index_buffer.push_back(index + 4);
               all_lines_index_buffer.push_back(index + 7);

               // y lines

               all_lines_index_buffer.push_back(index + 0);
               all_lines_index_buffer.push_back(index + 3);

               all_lines_index_buffer.push_back(index + 1);
               all_lines_index_buffer.push_back(index + 2);

               all_lines_index_buffer.push_back(index + 4);
               all_lines_index_buffer.push_back(index + 5);

               all_lines_index_buffer.push_back(index + 7);
               all_lines_index_buffer.push_back(index + 6);

               // z lines

               all_lines_index_buffer.push_back(index + 0);
               all_lines_index_buffer.push_back(index + 4);

               all_lines_index_buffer.push_back(index + 1);
               all_lines_index_buffer.push_back(index + 7);

               all_lines_index_buffer.push_back(index + 3);
               all_lines_index_buffer.push_back(index + 5);

               all_lines_index_buffer.push_back(index + 2);
               all_lines_index_buffer.push_back(index + 6);

               all_index_buffer.push_back(index + 0);
               all_index_buffer.push_back(index + 1);
               all_index_buffer.push_back(index + 2);

               all_index_buffer.push_back(index + 0);
               all_index_buffer.push_back(index + 2);
               all_index_buffer.push_back(index + 3);

               all_index_buffer.push_back(index + 4);
               all_index_buffer.push_back(index + 0);
               all_index_buffer.push_back(index + 3);

               all_index_buffer.push_back(index + 4);
               all_index_buffer.push_back(index + 3);
               all_index_buffer.push_back(index + 5);

               all_index_buffer.push_back(index + 7);
               all_index_buffer.push_back(index + 4);
               all_index_buffer.push_back(index + 5);

               all_index_buffer.push_back(index + 7);
               all_index_buffer.push_back(index + 5);
               all_index_buffer.push_back(index + 6);

               all_index_buffer.push_back(index + 1);
               all_index_buffer.push_back(index + 7);
               all_index_buffer.push_back(index + 6);

               all_index_buffer.push_back(index + 1);
               all_index_buffer.push_back(index + 6);
               all_index_buffer.push_back(index + 2);

               all_index_buffer.push_back(index + 3);
               all_index_buffer.push_back(index + 2);
               all_index_buffer.push_back(index + 6);

               all_index_buffer.push_back(index + 3);
               all_index_buffer.push_back(index + 6);
               all_index_buffer.push_back(index + 5);

               all_index_buffer.push_back(index + 0);
               all_index_buffer.push_back(index + 4);
               all_index_buffer.push_back(index + 7);

               all_index_buffer.push_back(index + 0);
               all_index_buffer.push_back(index + 7);
               all_index_buffer.push_back(index + 1);

               index += 8;

               // enqueue children, if any
               if (node->high != nullptr)
                  queue.push(std::make_pair(node->high, nodeDepth + 1));
               if (node->low != nullptr)
                  queue.push(std::make_pair(node->low, nodeDepth + 1));
            }
         }

         // all nodes
         gl::glGenVertexArrays(1, &all_vao_handle);
         gl::glGenVertexArrays(1, &all_vao_handle);
         gl::glBindVertexArray(all_vao_handle);

         gl::glGenBuffers(1, &all_index_buffer_handle);
         gl::glBindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, all_index_buffer_handle);
         gl::glBufferData(gl::GL_ELEMENT_ARRAY_BUFFER, all_index_buffer.size() * sizeof(float),
                          &all_index_buffer[0], gl::GL_STATIC_DRAW);


         gl::glGenBuffers(1, &all_vertex_buffer_handle);
         gl::glBindBuffer(gl::GL_ARRAY_BUFFER, all_vertex_buffer_handle);
         gl::glBufferData(gl::GL_ARRAY_BUFFER, all_vertex_buffer.size() * sizeof(float), &all_vertex_buffer[0],
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
         gl::glBindVertexArray(0);
         // selected node
         gl::glGenVertexArrays(1, &selected_vao_handle);
         gl::glBindVertexArray(selected_vao_handle);

         gl::glGenBuffers(1, &selected_node_index_handle);
         gl::glBindBuffer(gl::GL_ELEMENT_ARRAY_BUFFER, selected_node_index_handle);
         gl::glBufferData(gl::GL_ELEMENT_ARRAY_BUFFER, 24 * sizeof(float), &selected_index_buffer[0], gl::GL_STATIC_DRAW);

         gl::glGenBuffers(1, &selected_vbo_handle);
         gl::glBindBuffer(gl::GL_ARRAY_BUFFER, selected_vbo_handle);
         gl::glBufferData(gl::GL_ARRAY_BUFFER, 24 * sizeof(float), &selected_vertex_buffer[0], gl::GL_STATIC_DRAW);
         gl::glVertexAttribPointer(
               0,        // attribute 0 - must match layout in shader
               3,        // size
               gl::GL_FLOAT, // type
               gl::GL_FALSE, // normalized?
               0,  // stride
               (void*)0 // array buffer offset
         );
         gl::glEnableVertexAttribArray(0);
         gl::glBindVertexArray(0);

         select_node(bvh.root);
      }

      void draw_all(const glm::mat4 &mvp) {
         gl::glEnable(gl::GL_DEPTH_TEST);
         all_program->use();
         all_program->set_uniform(uniform_color, {1.0f, 1.0f, 1.0f, 0.06250f});
         all_program->set_uniform(uniform_mvp, mvp);
         gl::glBindVertexArray(all_vao_handle);
         gl::glDrawElements(gl::GL_LINES, all_lines_index_buffer.size(), gl::GL_UNSIGNED_INT, (void*)0);
      }

      void draw_selected(const glm::mat4 &mvp) {
         gl::glDisable(gl::GL_DEPTH_TEST);
         all_program->use();
         all_program->set_uniform(uniform_color, {1.0f, 1.0f, 1.0f, 0.5f});
         all_program->set_uniform(uniform_mvp, mvp);
         gl::glBindVertexArray(selected_vao_handle);
         gl::glDrawElements(gl::GL_LINES, 24, gl::GL_UNSIGNED_INT, (void*)0);
      }
   };
}