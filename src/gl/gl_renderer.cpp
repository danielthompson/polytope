//
// Created by daniel on 1/8/20.
//
#include "gl_renderer.h"
#include "../common/utilities/Common.h"
#include "bb_vao.h"

#define GLFW_INCLUDE_NONE
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/rotate_vector.hpp>
#include <fstream>
#include <sstream>
#include <queue>
#include <utility>

namespace poly {

   static void error_callback(int error, const char* description)
   {
      fprintf(stderr, "Error: %s\n", description);
   }

   poly::bvh_node* rootNode = nullptr;
   poly::bvh_node* currentNode = nullptr;

   gl::GLuint bbSelectedNodeVaoHandle;
   gl::GLuint bbSelectedNodeIndexBufferHandle;
   gl::GLuint bbSelectedNodeVertexBufferHandle;

   gl::GLsizei viewport_width;
   gl::GLsizei viewport_height;

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
            case GLFW_KEY_Z: {
               if (currentNode && currentNode->low) {
                  currentNode = currentNode->low;
                  Log.debug("Moving down to low node (left).");
                  renderer->bb.select_node(currentNode);
               }
               break;
            }
            case GLFW_KEY_X: {
               if (currentNode && currentNode->high) {
                  currentNode = currentNode->high;
                  Log.debug("Moving down to high node (right).");
                  renderer->bb.select_node(currentNode);
               }
               break;
            }
            case GLFW_KEY_R: {
               if (rootNode) {
                  currentNode = rootNode;
                  Log.debug("Resetting to root node.");
                  renderer->bb.select_node(currentNode);
               }
               break;
            }
            default: {
               // do nothing
            }
         }
      }
   }

   gl::GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path){

      // Create the shaders
      gl::GLuint VertexShaderID = gl::glCreateShader(gl::GL_VERTEX_SHADER);
      gl::GLuint FragmentShaderID = gl::glCreateShader(gl::GL_FRAGMENT_SHADER);

      // Read the Vertex Shader code from the file
      std::string VertexShaderCode;
      std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
      if (VertexShaderStream.is_open()){
         std::stringstream sstr;
         sstr << VertexShaderStream.rdbuf();
         VertexShaderCode = sstr.str();
         VertexShaderStream.close();
      } else{
         printf("Impossible to open %s. Are you in the right directory ? Don't forget to read the FAQ !\n", vertex_file_path);
         getchar();
         return 0;
      }

      // Read the Fragment Shader code from the file
      std::string FragmentShaderCode;
      std::ifstream FragmentShaderStream(fragment_file_path, std::ios::in);
      if(FragmentShaderStream.is_open()){
         std::stringstream sstr;
         sstr << FragmentShaderStream.rdbuf();
         FragmentShaderCode = sstr.str();
         FragmentShaderStream.close();
      }

      gl::GLint Result = 0;
      int InfoLogLength;

      // Compile Vertex Shader
      printf("Compiling shader : %s\n", vertex_file_path);
      char const * VertexSourcePointer = VertexShaderCode.c_str();
      gl::glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
      gl::glCompileShader(VertexShaderID);

      // Check Vertex Shader
      gl::glGetShaderiv(VertexShaderID, gl::GL_COMPILE_STATUS, &Result);
      gl::glGetShaderiv(VertexShaderID, gl::GL_INFO_LOG_LENGTH, &InfoLogLength);
      if ( InfoLogLength > 0 ){
         std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
         gl::glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
         printf("%s\n", &VertexShaderErrorMessage[0]);
      }

      // Compile Fragment Shader
      printf("Compiling shader : %s\n", fragment_file_path);
      char const * FragmentSourcePointer = FragmentShaderCode.c_str();
      gl::glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
      gl::glCompileShader(FragmentShaderID);

      // Check Fragment Shader
      gl::glGetShaderiv(FragmentShaderID, gl::GL_COMPILE_STATUS, &Result);
      gl::glGetShaderiv(FragmentShaderID, gl::GL_INFO_LOG_LENGTH, &InfoLogLength);
      if ( InfoLogLength > 0 ){
         std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
         gl::glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
         printf("%s\n", &FragmentShaderErrorMessage[0]);
      }

      // Link the program
      printf("Linking program\n");
      gl::GLuint ProgramID = gl::glCreateProgram();
      gl::glAttachShader(ProgramID, VertexShaderID);
      gl::glAttachShader(ProgramID, FragmentShaderID);
      gl::glLinkProgram(ProgramID);

      // Check the program
      gl::glGetProgramiv(ProgramID, gl::GL_LINK_STATUS, &Result);
      gl::glGetProgramiv(ProgramID, gl::GL_INFO_LOG_LENGTH, &InfoLogLength);
      if ( InfoLogLength > 0 ){
         std::vector<char> ProgramErrorMessage(InfoLogLength+1);
         gl::glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
         printf("%s\n", &ProgramErrorMessage[0]);
      }

      gl::glDetachShader(ProgramID, VertexShaderID);
      gl::glDetachShader(ProgramID, FragmentShaderID);

      gl::glDeleteShader(VertexShaderID);
      gl::glDeleteShader(FragmentShaderID);

      return ProgramID;
   }

   float currentXpos, currentYpos;

   glm::vec3 eye, lookAt, up;

   float fov;

   glm::mat4 viewMatrix;
   glm::mat4 projectionMatrix;

   bool rightPressed = false;

   void cursor_position_callback(GLFWwindow* window, const double xpos, const double ypos)
   {
      if (!rightPressed) {
         currentXpos = xpos;
         currentYpos = ypos;
         return;
      }

      const float distance = glm::distance(eye, lookAt);

      const float dtheta = (currentXpos - (float)xpos) * .25f;
      const float dphi = ((float)ypos - currentYpos) * .25f;

      glm::vec3 dir = glm::normalize(eye - lookAt);
      glm::vec3 thetaDir = glm::rotate(dir, glm::radians(dtheta), up);
      eye = lookAt + distance * thetaDir;

      dir = glm::normalize(eye - lookAt);
      glm::vec3 right = glm::normalize(glm::cross(dir, up));
      glm::vec3 phiDir = glm::rotate(dir, glm::radians(dphi), right);
      eye = lookAt + distance * phiDir;

      up = glm::cross(right, phiDir);

      currentXpos = xpos;
      currentYpos = ypos;
   }

   void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
   {
      auto renderer = static_cast<gl_renderer *>(glfwGetWindowUserPointer(window));
      switch (button) {
         case GLFW_MOUSE_BUTTON_LEFT: {
            if (action != GLFW_PRESS)
               break;
            
            // get first intersection at clicked pixel
            poly::Point2f pixel = {currentXpos / (float)viewport_width, currentYpos / (float)viewport_height};
            poly::Ray camera_ray = renderer->scene->Camera->get_ray_for_ndc(pixel);
            
            poly::Intersection intersection = renderer->scene->intersect(camera_ray, currentXpos, currentYpos);
            
            // set cursor position to intersection location
            if (intersection.Hits) {
               lookAt.x = intersection.Location.x;
               lookAt.y = intersection.Location.y;
               lookAt.z = intersection.Location.z;
            }
            
            // draw cursor
            
            break;
         }
         case GLFW_MOUSE_BUTTON_MIDDLE: {
            std::ostringstream str;
            if (action == GLFW_PRESS) {
               rightPressed = true;
               str << "Right mouse press @ " << currentXpos << ", " << currentYpos;
            } else if (action == GLFW_RELEASE) {
               rightPressed = false;
               str << "Right mouse release @ " << currentXpos << ", " << currentYpos;
            }
            Log.debug(str.str());
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

   gl_renderer::gl_renderer(std::shared_ptr<poly::scene> source_scene) : scene(source_scene) {
      // glfw init

      viewport_width = 1600;
      viewport_height = 1000;
      {
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

         window = glfwCreateWindow(viewport_width, viewport_height, "Polytope", NULL, NULL);
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
      }
      // end GLFW init
   }
   
   void gl_renderer::render()
   {
      bb.init(scene->bvh_root);
      
      // actual drawing code
      for (const auto& mesh : scene->Shapes) {
         const unsigned int indices = mesh->mesh_geometry->num_faces * 3;

         rootNode = scene->bvh_root.root;
         currentNode = rootNode;

         std::vector<unsigned int> shapeIndexVector(indices, 0);
         std::vector<float> shapeVertexVector(indices * 3, 0.f);
         std::vector<float> shapeNormalVector(indices * 3, 0.f);

         for (unsigned int i = 0; i < mesh->mesh_geometry->num_faces; i++) {

            std::shared_ptr<poly::mesh_geometry> geometry = mesh->mesh_geometry;

            poly::Point p = {geometry->x_packed[geometry->fv0[i]],
                             geometry->y_packed[geometry->fv0[i]],
                             geometry->z_packed[geometry->fv0[i]]
            };
            mesh->object_to_world->ApplyInPlace(p);
            shapeVertexVector[9 * i] = p.x;
            shapeVertexVector[9 * i + 1] = p.y;
            shapeVertexVector[9 * i + 2] = p.z;

            p = {geometry->x_packed[geometry->fv1[i]],
                 geometry->y_packed[geometry->fv1[i]],
                 geometry->z_packed[geometry->fv1[i]]
            };
            mesh->object_to_world->ApplyInPlace(p);
            shapeVertexVector[9 * i + 3] = p.x;
            shapeVertexVector[9 * i + 4] = p.y;
            shapeVertexVector[9 * i + 5] = p.z;

            p = {geometry->x_packed[geometry->fv2[i]],
                 geometry->y_packed[geometry->fv2[i]],
                 geometry->z_packed[geometry->fv2[i]]
            };
            mesh->object_to_world->ApplyInPlace(p);
            shapeVertexVector[9 * i + 6] = p.x;
            shapeVertexVector[9 * i + 7] = p.y;
            shapeVertexVector[9 * i + 8] = p.z;

//         poly::Normal n = { geometry->nx_packed[geometry->fv0[i]],
//                            geometry->ny_packed[geometry->fv0[i]],
//                            geometry->nz_packed[geometry->fv0[i]]
//         };
            poly::Normal n = {0, 0, 0};

            mesh->object_to_world->ApplyInPlace(n);
            shapeNormalVector[9 * i] = n.x;
            shapeNormalVector[9 * i + 1] = n.y;
            shapeNormalVector[9 * i + 2] = n.z;

//         n = {geometry->nx_packed[geometry->fv1[i]],
//              geometry->ny_packed[geometry->fv1[i]],
//              geometry->nz_packed[geometry->fv1[i]]
//         };
            n = {0, 0, 0};
            mesh->object_to_world->ApplyInPlace(n);
            shapeNormalVector[9 * i + 3] = n.x;
            shapeNormalVector[9 * i + 4] = n.y;
            shapeNormalVector[9 * i + 5] = n.z;

//         n = { geometry->nx_packed[geometry->fv2[i]],
//               geometry->ny_packed[geometry->fv2[i]],
//               geometry->nz_packed[geometry->fv2[i]]
//         };
            n = {0, 0, 0};
            mesh->object_to_world->ApplyInPlace(n);
            shapeNormalVector[9 * i + 6] = n.x;
            shapeNormalVector[9 * i + 7] = n.y;
            shapeNormalVector[9 * i + 8] = n.z;

            shapeIndexVector[3 * i] = 3 * i;
            shapeIndexVector[3 * i + 1] = 3 * i + 1;
            shapeIndexVector[3 * i + 2] = 3 * i + 2;
         }



         // selected BB stuff

         gl::GLuint shapeVao;
         gl::glGenVertexArrays(1, &shapeVao);
         gl::glBindVertexArray(shapeVao);

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
      
      // Ensure we can capture the escape key being pressed below
      glfwSetInputMode(window, GLFW_STICKY_KEYS, (int)gl::GL_TRUE);

      gl::GLuint shapeProgramHandle = LoadShaders("../src/gl/shape/vert.glsl", "../src/gl/shape/frag.glsl");
      gl::GLuint bboxProgramHandle = LoadShaders("../src/gl/bbox/vert.glsl", "../src/gl/bbox/frag.glsl");

      // projection matrix - 45deg fov, 4:3 ratio, display range - 0.1 <-> 100 units
      fov = scene->Camera->Settings.FieldOfView;
      projectionMatrix = glm::perspective(glm::radians(fov), (float)viewport_width / (float)viewport_height, 0.1f, 100.0f);

      eye = glm::vec3(scene->Camera->eye.x, scene->Camera->eye.y, scene->Camera->eye.z);
      lookAt = glm::vec3(scene->Camera->lookAt.x, scene->Camera->lookAt.y, -scene->Camera->lookAt.z);
      up = glm::vec3(scene->Camera->up.x, scene->Camera->up.y, scene->Camera->up.z);

      // camera matrix

      // model - identity, since model will be at the origin for now
      glm::mat4 model = glm::mat4(1.0f);

      // get uniform handle
      gl::GLuint matrixID = gl::glGetUniformLocation(shapeProgramHandle, "mvp");
      gl::GLuint colorInId = gl::glGetUniformLocation(shapeProgramHandle, "colorIn");

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
      do {
         // clear the screen
         gl::glClear(gl::GL_COLOR_BUFFER_BIT | gl::GL_DEPTH_BUFFER_BIT);

         // calculate new mvp
         projectionMatrix = glm::perspective(glm::radians(fov), (float)viewport_width / (float)viewport_height, 0.1f, 100.0f);

         viewMatrix = glm::lookAt(eye, lookAt, up);
         glm::mat4 mvp = projectionMatrix * viewMatrix * model;

         // bounding boxes
         gl::glUseProgram(bboxProgramHandle);
         
         // send transformation matrix to the currently bound shader, in the "mvp" uniform
         gl::glUniformMatrix4fv(matrixID, 1, gl::GL_FALSE, &mvp[0][0]);

         gl::glEnable(gl::GL_DEPTH_TEST);
         //glEnable(GL_BLEND);

         gl::glUniform4f(colorInId, 1.0f, 1.0f, 1.0f, 0.06250f);
         bb.draw_all();
         
         //glDisable(GL_DEPTH_TEST);
         //glEnable(GL_BLEND);
//
//         // shapes
//         gl::glUseProgram(shapeProgramHandle);
//         gl::glBindVertexArray(shapeVao);
//
//         // send transformation matrix to the currently bound shader, in the "mvp" uniform
//         gl::glUniformMatrix4fv(matrixID, 1, gl::GL_FALSE, &mvp[0][0]);
//
//         gl::glEnable(gl::GL_DEPTH_TEST);
//         gl::glDisable(gl::GL_BLEND);
//
//         gl::glUniform4f(colorInId, 0.4f, 0.5f, 0.5f, 0.150f);
//         gl::glPolygonMode(gl::GL_FRONT_AND_BACK, gl::GL_FILL);
//         gl::glDrawElements(gl::GL_TRIANGLES, shapeIndexVector.size(), gl::GL_UNSIGNED_INT, (void*)0);
//
//         gl::glDisable(gl::GL_DEPTH_TEST);
//         gl::glEnable(gl::GL_BLEND);
//
//         gl::glUniform4f(colorInId, 1.0f, 1.0f, 1.0f, 0.06250f);
//         gl::glPolygonMode(gl::GL_FRONT_AND_BACK, gl::GL_LINE);
//         gl::glDrawElements(gl::GL_TRIANGLES, shapeIndexVector.size(), gl::GL_UNSIGNED_INT, (void*)0);

         // bounding boxes - all
         gl::glUseProgram(bboxProgramHandle);
         
         // send transformation matrix to the currently bound shader, in the "mvp" uniform
         gl::glUniformMatrix4fv(matrixID, 1, gl::GL_FALSE, &mvp[0][0]);

         gl::glDisable(gl::GL_DEPTH_TEST);
         gl::glEnable(gl::GL_BLEND);

         gl::glUniform4f(colorInId, 1.0f, 1.0f, 1.0f, 0.006250f);
         bb.draw_all();

         // bounding boxes - selected
         gl::glBindVertexArray(bbSelectedNodeVaoHandle);

         // send transformation matrix to the currently bound shader, in the "mvp" uniform
         gl::glUniformMatrix4fv(matrixID, 1, gl::GL_FALSE, &mvp[0][0]);

         gl::glDisable(gl::GL_DEPTH_TEST);
         //glEnable(GL_BLEND);

         gl::glUniform4f(colorInId, 1.0f, 1.0f, 1.0f, 0.5f);
         bb.draw_selected();

         // swap bufffers
         glfwSwapBuffers(window);
         glfwPollEvents();

      } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

      glfwDestroyWindow(window);

      glfwTerminate();
      //exit(EXIT_SUCCESS);
   }

   
}