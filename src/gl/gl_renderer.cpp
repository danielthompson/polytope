//
// Created by daniel on 1/8/20.
//
#include "gl_renderer.h"
#include "../common/utilities/Common.h"
#include "bb_vao.h"
#include "mesh_vao.h"
#include "line/line.h"
#include "gl_recorder.h"

#define GLFW_INCLUDE_NONE
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>
#include <fstream>
#include <sstream>
#include <queue>

namespace poly {

   std::unique_ptr<poly::gl_recorder> recorder;
   
   static void error_callback(int error, const char* description)
   {
      fprintf(stderr, "Error: %s\n", description);
   }

   poly::bvh_node* rootNode = nullptr;
   poly::bvh_node* currentNode = nullptr;

   gl::GLsizei viewport_width;
   gl::GLsizei viewport_height;

   bool spin = false;
   
   static void framebuffer_size_callback(GLFWwindow* window, int width, int height)
   {
      viewport_width = width;
      viewport_height = height;
      gl::glViewport(0, 0, viewport_width, viewport_height);
   }

   static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
   {

      auto renderer = static_cast<gl_renderer *>(glfwGetWindowUserPointer(window));
      
      if (action == GLFW_PRESS) {
         switch (key) {
            case GLFW_KEY_A: {
               recorder->start();
               break;
            }
            case GLFW_KEY_Z: {
               recorder->stop();
               break;
            }
            case GLFW_KEY_ESCAPE: {
               glfwSetWindowShouldClose(window, GLFW_TRUE);
               break;
            }
            case GLFW_KEY_S: {
//               if (currentNode && currentNode->parent) {
//                  currentNode = currentNode->parent;
//                  Log.WithTime("Moving up to parent node.");
//                  selectNode(currentNode);
//               }
//               else {
//                  if (currentNode == rootNode)
//                     Log.WithTime("Can't move to parent (already at root).");
//                  else
//                     Log.WithTime("Can't move to parent, but not at root either (probably a bug). :/");
//               }
               break;
            }
            case GLFW_KEY_J: {
               if (currentNode && currentNode->low) {
                  currentNode = currentNode->low;
                  LOG_DEBUG("Moving down to low node (left).");
                  renderer->bb.select_node(currentNode);
               }
               break;
            }
            case GLFW_KEY_K: {
               if (currentNode && currentNode->high) {
                  currentNode = currentNode->high;
                  LOG_DEBUG("Moving down to high node (right).");
                  renderer->bb.select_node(currentNode);
               }
               break;
            }
            case GLFW_KEY_U: {
               if (rootNode) {
                  currentNode = rootNode;
                  LOG_DEBUG("Resetting to root node.");
                  renderer->bb.select_node(currentNode);
               }
               break;
            }

            case GLFW_KEY_Q: {
               spin = !spin;
               break;
            }
            
            default: {
               // do nothing
            }
         }
      }
   }

   float currentXpos, currentYpos;

   glm::vec3 eye, lookAt, up;

   float fov;

   glm::mat4 viewMatrix;
   glm::mat4 projectionMatrix;

   poly::Sample most_recent_sample;
   
   std::vector<poly::line> lines = { };
   
   bool rightPressed = false;

   void update_orientation(const float dtheta, const float dphi) {
      const float distance = glm::distance(eye, lookAt);

      glm::vec3 dir = glm::normalize(eye - lookAt);
      glm::vec3 thetaDir = glm::rotate(dir, -glm::radians(dtheta), up);
      eye = lookAt + distance * thetaDir;

      dir = glm::normalize(eye - lookAt);
      glm::vec3 right = glm::normalize(glm::cross(dir, up));
      glm::vec3 phiDir = glm::rotate(dir, glm::radians(dphi), right);
      eye = lookAt + distance * phiDir;

      up = glm::cross(right, phiDir);
   }
   
   void cursor_position_callback(GLFWwindow* window, const double xpos, const double ypos)
   {
      if (!rightPressed) {
         currentXpos = xpos;
         currentYpos = ypos;
         return;
      }

      
      const float dtheta = (currentXpos - (float)xpos) * .25f;
      const float dphi = ((float)ypos - currentYpos) * .25f;

      update_orientation(dtheta, dphi);

      currentXpos = xpos;
      currentYpos = ypos;
   }

   void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
   {
      auto renderer = static_cast<poly::gl_renderer *>(glfwGetWindowUserPointer(window));
      switch (button) {
         case GLFW_MOUSE_BUTTON_LEFT: {
            if (action != GLFW_PRESS)
               break;
            
            // get first intersection at clicked pixel
            poly::point2f pixel = {currentXpos / (float)viewport_width, currentYpos / (float)viewport_height};

            poly::ray camera_ray = renderer->runner->Scene->Camera->get_ray_for_ndc(pixel);
            LOG_DEBUG("camera ray: " << camera_ray);

            lines.clear();
            
            poly::point prev_location = {camera_ray.origin};
            poly::vector prev_dir = {camera_ray.direction};
            
            most_recent_sample = renderer->runner->integrator->get_sample(camera_ray, 0, currentXpos, currentYpos);

            for (int i = 0; i < most_recent_sample.intersections.size(); i++) {
               auto& element = most_recent_sample.intersections[i];
               
               if (element.Hits) {
                  glm::vec4 white = {1.f, 1.f, 1.f, 1.f};
                  lines.emplace_back(prev_location, element.location, white );
                  LOG_DEBUG("Bounce " << i << ": hit");
                  LOG_DEBUG("  t        " << i << ": " << element.t);
                  LOG_DEBUG("  location " << i << ": " << element.location);
                  LOG_DEBUG("  b normal " << i << ": " << element.bent_normal);
                  LOG_DEBUG("  g normal " << i << ": " << element.geo_normal);
                  LOG_DEBUG("  outgoing " << i << ": " << element.outgoing);
                  LOG_DEBUG("  mesh_index " << element.mesh_index);
                  LOG_DEBUG("  face_index " << element.face_index);
                  prev_location = element.location;
                  prev_dir = element.outgoing;
               }
               else {
                  glm::vec4 reddish = {1.f, 0.5f, 0.5f, 1.f};
                  poly::point endpoint = prev_location + (prev_dir * 10); 
                  lines.emplace_back(prev_location, endpoint, reddish );
                  LOG_DEBUG("Bounce " << i << ": miss");
               }
            }
            
            break;
         }
         case GLFW_MOUSE_BUTTON_MIDDLE: {
            if (action == GLFW_PRESS) {
               rightPressed = true;
               LOG_DEBUG("Right mouse press @ " << currentXpos << ", " << currentYpos);
            } else if (action == GLFW_RELEASE) {
               rightPressed = false;
               LOG_DEBUG("Right mouse release @ " << currentXpos << ", " << currentYpos);
            }
            break;
         }
      }
   }

   void scroll_callback(GLFWwindow* window, const double xoffset, const double yoffset)
   {
      fov -= (float)yoffset;
      if (fov < 0) {
         fov = 0;
      }
      else if (fov > 180) {
         fov = 180;
      }
   }

   gl_renderer::gl_renderer(std::shared_ptr<poly::runner> runner) : runner(runner) {
      viewport_width = runner->Bounds.x;
      viewport_height = runner->Bounds.y;
   
      glfwSetErrorCallback(error_callback);

      if (!glfwInit()) {
         fprintf(stderr, "Failed to initialize GLFW :/");
         exit(EXIT_FAILURE);
      }

      glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
      glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
      glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
      glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

      glfwWindowHint(GLFW_SAMPLES, 16);

      window = glfwCreateWindow(viewport_width, viewport_height, "Polytope", nullptr, nullptr);
      if (!window) {
         fprintf(stderr, "Failed to open GLFW window :/");
         glfwTerminate();
         exit(EXIT_FAILURE);
      }

      glfwSetKeyCallback(window, key_callback);

      glfwMakeContextCurrent(window);
      glbinding::initialize(glfwGetProcAddress);
      glfwSwapInterval(1);

      glfwSetCursorPosCallback(window, cursor_position_callback);
      glfwSetMouseButtonCallback(window, mouse_button_callback);
      glfwSetScrollCallback(window, scroll_callback);
      glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
      glfwSetWindowUserPointer(window, this);
      
      // Ensure we can capture the escape key being pressed below
      glfwSetInputMode(window, GLFW_STICKY_KEYS, (int)gl::GL_TRUE);
      
      recorder = std::make_unique<poly::gl_recorder>(viewport_width, viewport_height);
   }
   
   void gl_renderer::render()
   {
      bb.init(runner->Scene->bvh_root);
      rootNode = runner->Scene->bvh_root.root;
      currentNode = rootNode;
      
      std::vector<poly::mesh_vao> mesh_vaos;
      mesh_vaos.reserve(runner->Scene->Shapes.size());

      // TODO do proper instancing
      for (int mesh_index = 0; mesh_index < runner->Scene->Shapes.size(); mesh_index++) {
         mesh_vaos.emplace_back();
         mesh_vaos[mesh_index].init(runner->Scene->Shapes[mesh_index]);
      }
      
      // projection matrix - 45deg fov, 4:3 ratio, display range - 0.1 <-> 100 units
      fov = runner->Scene->Camera->settings.field_of_view;
      projectionMatrix = glm::perspective(glm::radians(fov), (float)viewport_width / (float)viewport_height, 0.1f, 100.0f);
      
      eye = glm::vec3(runner->Scene->Camera->eye.x, runner->Scene->Camera->eye.y, runner->Scene->Camera->eye.z);
      lookAt = glm::vec3(runner->Scene->Camera->lookAt.x, runner->Scene->Camera->lookAt.y, -runner->Scene->Camera->lookAt.z);
      up = glm::vec3(runner->Scene->Camera->up.x, runner->Scene->Camera->up.y, runner->Scene->Camera->up.z);

      // camera matrix

      // model - identity, since model will be at the origin for now
      glm::mat4 model = glm::mat4(1.0f);

      gl::glEnable(gl::GL_DEPTH_TEST);
      gl::glDepthFunc(gl::GL_LESS);

      gl::glEnable(gl::GL_BLEND);
      gl::glBlendFunc(gl::GL_SRC_ALPHA, gl::GL_ONE_MINUS_SRC_ALPHA);

      //glEnable(GL_CULL_FACE);
      gl::glDepthFunc(gl::GL_LESS);
      gl::glBlendFunc(gl::GL_SRC_ALPHA, gl::GL_ONE_MINUS_SRC_ALPHA);

      gl::glEnable(gl::GL_MULTISAMPLE);

      gl::glClearColor(0.15f, 0.15f, 0.15f, 0.0f);
      gl::glEnable(gl::GL_BLEND);
      glm::vec4 x_scale = {-1, 1, 1, 1};

      do {
         // clear the screen
         gl::glClear(gl::GL_COLOR_BUFFER_BIT | gl::GL_DEPTH_BUFFER_BIT);

         if (spin) {
            update_orientation(0.5f, 0.f);
         }
         
         // calculate new mvp
         projectionMatrix = glm::perspective(glm::radians(fov), (float)viewport_width / (float)viewport_height, 0.1f, 100.0f);
         projectionMatrix *= x_scale;

         viewMatrix = glm::lookAt(eye, lookAt, up);
         glm::mat4 mvp = projectionMatrix * viewMatrix * model;
        // bb.draw_all(mvp);
         
         for (auto & mesh_vao : mesh_vaos) {
            mesh_vao.draw(mvp);
         }

         bb.draw_all(mvp);
         bb.draw_selected(mvp);

         for (const auto& element : lines) {
            element.draw(mvp);   
         }

         recorder->capture_frame();
         
         // swap bufffers
         glfwSwapBuffers(window);
         glfwPollEvents();

      } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

      recorder->stop();
      
      glfwDestroyWindow(window);

      glfwTerminate();
   }
}