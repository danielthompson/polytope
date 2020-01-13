//
// Created by daniel on 1/8/20.
//

#include "GLRenderer.h"
#include "../shapes/TriangleMesh.h"
#include "../utilities/Common.h"

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

namespace Polytope {

   static void error_callback(int error, const char* description)
   {
      fprintf(stderr, "Error: %s\n", description);
   }

   static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
   {
      if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
         glfwSetWindowShouldClose(window, GLFW_TRUE);
   }

   using namespace gl;


   glm::mat4 MakeMat4(const Polytope::Transform &transform) {
      float arr[16];
      arr[0] = -transform.Matrix.Matrix[0][0];
      arr[1] = transform.Matrix.Matrix[1][0];
      arr[2] = -transform.Matrix.Matrix[2][0];
      arr[3] = transform.Matrix.Matrix[3][0];

      arr[4] = -transform.Matrix.Matrix[0][1];
      arr[5] = transform.Matrix.Matrix[1][1];
      arr[6] = -transform.Matrix.Matrix[2][1];
      arr[7] = transform.Matrix.Matrix[3][1];

      arr[8] = transform.Matrix.Matrix[0][2];
      arr[9] = transform.Matrix.Matrix[1][2];
      arr[10] = -transform.Matrix.Matrix[2][2];
      arr[11] = transform.Matrix.Matrix[3][2];

      arr[12] = transform.Matrix.Matrix[0][3];
      arr[13] = transform.Matrix.Matrix[1][3];
      arr[14] = -transform.Matrix.Matrix[2][3];
      arr[15] = transform.Matrix.Matrix[3][3];

      glm::mat4 mat = glm::make_mat4(arr);
      return mat;
   }

   GLuint LoadShaders(const char * vertex_file_path, const char * fragment_file_path){

      // Create the shaders
      GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
      GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

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

      GLint Result = 0;
      int InfoLogLength;

      // Compile Vertex Shader
      printf("Compiling shader : %s\n", vertex_file_path);
      char const * VertexSourcePointer = VertexShaderCode.c_str();
      glShaderSource(VertexShaderID, 1, &VertexSourcePointer , NULL);
      glCompileShader(VertexShaderID);

      // Check Vertex Shader
      glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
      glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
      if ( InfoLogLength > 0 ){
         std::vector<char> VertexShaderErrorMessage(InfoLogLength+1);
         glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
         printf("%s\n", &VertexShaderErrorMessage[0]);
      }

      // Compile Fragment Shader
      printf("Compiling shader : %s\n", fragment_file_path);
      char const * FragmentSourcePointer = FragmentShaderCode.c_str();
      glShaderSource(FragmentShaderID, 1, &FragmentSourcePointer , NULL);
      glCompileShader(FragmentShaderID);

      // Check Fragment Shader
      glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
      glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
      if ( InfoLogLength > 0 ){
         std::vector<char> FragmentShaderErrorMessage(InfoLogLength+1);
         glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL, &FragmentShaderErrorMessage[0]);
         printf("%s\n", &FragmentShaderErrorMessage[0]);
      }

      // Link the program
      printf("Linking program\n");
      GLuint ProgramID = glCreateProgram();
      glAttachShader(ProgramID, VertexShaderID);
      glAttachShader(ProgramID, FragmentShaderID);
      glLinkProgram(ProgramID);

      // Check the program
      glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
      glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
      if ( InfoLogLength > 0 ){
         std::vector<char> ProgramErrorMessage(InfoLogLength+1);
         glGetProgramInfoLog(ProgramID, InfoLogLength, NULL, &ProgramErrorMessage[0]);
         printf("%s\n", &ProgramErrorMessage[0]);
      }

      glDetachShader(ProgramID, VertexShaderID);
      glDetachShader(ProgramID, FragmentShaderID);

      glDeleteShader(VertexShaderID);
      glDeleteShader(FragmentShaderID);

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
      if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
         std::ostringstream str;
         if (action == GLFW_PRESS) {
            rightPressed = true;
            str << "Right mouse press @ " << currentXpos << ", " << currentYpos;
         }
         else if (action == GLFW_RELEASE) {
            rightPressed = false;
            str << "Right mouse release @ " << currentXpos << ", " << currentYpos;
         }
         Log.WithTime(str.str());
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

   void GLRenderer::Render(Polytope::AbstractScene* scene)
   {
      // glfw init

      constexpr int width = 720, height = 480;

      GLFWwindow* window;

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

         window = glfwCreateWindow(1600, 1000, "Simple example", NULL, NULL);
         if (!window) {
            fprintf(stderr, "Failed to open GLFW window :/");
            glfwTerminate();
            exit(EXIT_FAILURE);
         }

         glfwSetKeyCallback(window, key_callback);

         glfwMakeContextCurrent(window);
         glbinding::initialize(glfwGetProcAddress);
         glfwSwapInterval(1);

         //glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
         // not supported on osx in 3.3
         // not supported in 3.2, which is latest available in ubuntu 18.04 LTS
//      if (glfwRawMouseMotionSupported()) {
//         Log.WithTime("GLFW: Raw mouse motion is supported, enabling...");
//         glfwSetInputMode(window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);
//      }
//      else {
//         Log.WithTime("GLFW: Raw mouse motion is not supported :/");
//      }
         glfwSetCursorPosCallback(window, cursor_position_callback);
         glfwSetMouseButtonCallback(window, mouse_button_callback);
         glfwSetScrollCallback(window, scroll_callback);
      }
      // end GLFW init

      // actual drawing code



      Polytope::TriangleMesh* mesh = (Polytope::TriangleMesh *)(scene->Shapes[0]);
      const unsigned int indices = mesh->Faces.size() * 3;

      std::vector<unsigned int> shapeIndexVector(indices, 0);
      std::vector<float> shapeVertexVector(indices * 3, 0.f);
      std::vector<float> shapeNormalVector(indices * 3, 0.f);

      for (unsigned int i = 0; i < mesh->Faces.size(); i++) {

         const Polytope::Point3ui face = mesh->Faces[i];
         shapeVertexVector[9 * i] = mesh->Vertices[face.x].x;
         shapeVertexVector[9 * i + 1] = mesh->Vertices[face.x].y;
         shapeVertexVector[9 * i + 2] = mesh->Vertices[face.x].z;

         shapeVertexVector[9 * i + 3] = mesh->Vertices[face.y].x;
         shapeVertexVector[9 * i + 4] = mesh->Vertices[face.y].y;
         shapeVertexVector[9 * i + 5] = mesh->Vertices[face.y].z;

         shapeVertexVector[9 * i + 6] = mesh->Vertices[face.z].x;
         shapeVertexVector[9 * i + 7] = mesh->Vertices[face.z].y;
         shapeVertexVector[9 * i + 8] = mesh->Vertices[face.z].z;

         shapeNormalVector[9 * i] = mesh->Normals[face.x].x;
         shapeNormalVector[9 * i + 1] = mesh->Normals[face.x].y;
         shapeNormalVector[9 * i + 2] = mesh->Normals[face.x].z;

         shapeNormalVector[9 * i + 3] = mesh->Normals[face.y].x;
         shapeNormalVector[9 * i + 4] = mesh->Normals[face.y].y;
         shapeNormalVector[9 * i + 5] = mesh->Normals[face.y].z;

         shapeNormalVector[9 * i + 6] = mesh->Normals[face.z].x;
         shapeNormalVector[9 * i + 7] = mesh->Normals[face.z].y;
         shapeNormalVector[9 * i + 8] = mesh->Normals[face.z].z;

         shapeIndexVector[3 * i] = 3 * i;
         shapeIndexVector[3 * i + 1] = 3 * i + 1;
         shapeIndexVector[3 * i + 2] = 3 * i + 2;
      }

      std::vector<unsigned int> bbIndexVector;
      std::vector<float> bbVertexVector;

      std::vector<unsigned int> bbLinesIndexVector;

      std::queue<Polytope::BVHNode*> queue;

      if (mesh->root != nullptr) {
         queue.push(mesh->root);

         unsigned int index = 0;

         while (!queue.empty()) {
            const BVHNode* node = queue.front();
            queue.pop();

            const Point low = node->bbox.p0;
            const Point high = node->bbox.p1;

            // add BB box to shapes
            bbVertexVector.push_back(low.x);
            bbVertexVector.push_back(low.y);
            bbVertexVector.push_back(low.z);

            bbVertexVector.push_back(high.x);
            bbVertexVector.push_back(low.y);
            bbVertexVector.push_back(low.z);

            bbVertexVector.push_back(high.x);
            bbVertexVector.push_back(high.y);
            bbVertexVector.push_back(low.z);

            bbVertexVector.push_back(low.x);
            bbVertexVector.push_back(high.y);
            bbVertexVector.push_back(low.z);

            bbVertexVector.push_back(low.x);
            bbVertexVector.push_back(low.y);
            bbVertexVector.push_back(high.z);

            bbVertexVector.push_back(low.x);
            bbVertexVector.push_back(high.y);
            bbVertexVector.push_back(high.z);

            bbVertexVector.push_back(high.x);
            bbVertexVector.push_back(high.y);
            bbVertexVector.push_back(high.z);

            bbVertexVector.push_back(high.x);
            bbVertexVector.push_back(low.y);
            bbVertexVector.push_back(high.z);

            // x lines

            bbLinesIndexVector.push_back(index + 0);
            bbLinesIndexVector.push_back(index + 1);

            bbLinesIndexVector.push_back(index + 3);
            bbLinesIndexVector.push_back(index + 2);

            bbLinesIndexVector.push_back(index + 5);
            bbLinesIndexVector.push_back(index + 6);

            bbLinesIndexVector.push_back(index + 4);
            bbLinesIndexVector.push_back(index + 7);

            // y lines

            bbLinesIndexVector.push_back(index + 0);
            bbLinesIndexVector.push_back(index + 3);

            bbLinesIndexVector.push_back(index + 1);
            bbLinesIndexVector.push_back(index + 2);

            bbLinesIndexVector.push_back(index + 4);
            bbLinesIndexVector.push_back(index + 5);

            bbLinesIndexVector.push_back(index + 7);
            bbLinesIndexVector.push_back(index + 6);

            // z lines

            bbLinesIndexVector.push_back(index + 0);
            bbLinesIndexVector.push_back(index + 4);

            bbLinesIndexVector.push_back(index + 1);
            bbLinesIndexVector.push_back(index + 7);

            bbLinesIndexVector.push_back(index + 3);
            bbLinesIndexVector.push_back(index + 5);

            bbLinesIndexVector.push_back(index + 2);
            bbLinesIndexVector.push_back(index + 6);

            bbIndexVector.push_back(index + 0);
            bbIndexVector.push_back(index + 1);
            bbIndexVector.push_back(index + 2);

            bbIndexVector.push_back(index + 0);
            bbIndexVector.push_back(index + 2);
            bbIndexVector.push_back(index + 3);

            bbIndexVector.push_back(index + 4);
            bbIndexVector.push_back(index + 0);
            bbIndexVector.push_back(index + 3);

            bbIndexVector.push_back(index + 4);
            bbIndexVector.push_back(index + 3);
            bbIndexVector.push_back(index + 5);

            bbIndexVector.push_back(index + 7);
            bbIndexVector.push_back(index + 4);
            bbIndexVector.push_back(index + 5);

            bbIndexVector.push_back(index + 7);
            bbIndexVector.push_back(index + 5);
            bbIndexVector.push_back(index + 6);

            bbIndexVector.push_back(index + 1);
            bbIndexVector.push_back(index + 7);
            bbIndexVector.push_back(index + 6);

            bbIndexVector.push_back(index + 1);
            bbIndexVector.push_back(index + 6);
            bbIndexVector.push_back(index + 2);

            bbIndexVector.push_back(index + 3);
            bbIndexVector.push_back(index + 2);
            bbIndexVector.push_back(index + 6);

            bbIndexVector.push_back(index + 3);
            bbIndexVector.push_back(index + 6);
            bbIndexVector.push_back(index + 5);

            bbIndexVector.push_back(index + 0);
            bbIndexVector.push_back(index + 4);
            bbIndexVector.push_back(index + 7);

            bbIndexVector.push_back(index + 0);
            bbIndexVector.push_back(index + 7);
            bbIndexVector.push_back(index + 1);

            index += 8;

            // enqueue children, if any
            if (node->high != nullptr)
               queue.push(node->high);
            if (node->low != nullptr)
               queue.push(node->low);
         }
      }

      GLuint shapeVao;
      glGenVertexArrays(1, &shapeVao);
      glBindVertexArray(shapeVao);

      // buffer for vertex indices
      GLuint shapeIndexBuffer;
      glGenBuffers(1, &shapeIndexBuffer);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, shapeIndexBuffer);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices * sizeof(indices), &shapeIndexVector[0], GL_STATIC_DRAW);

      // buffer for vertex locations
      GLuint shapeVertexBuffer;
      glGenBuffers(1, &shapeVertexBuffer);
      glBindBuffer(GL_ARRAY_BUFFER, shapeVertexBuffer);
      glBufferData(GL_ARRAY_BUFFER, indices * 3 * sizeof(indices), &shapeVertexVector[0], GL_STATIC_DRAW);
      glVertexAttribPointer(
            0,        // attribute 0 - must match layout in shader
            3,        // size
            GL_FLOAT, // type
            GL_FALSE, // normalized?
            0,  // stride
            (void*)0 // array buffer offset
      );
      glEnableVertexAttribArray(0);

      GLuint shapeNormalBuffer;
      glGenBuffers(1, &shapeNormalBuffer);
      glBindBuffer(GL_ARRAY_BUFFER, shapeNormalBuffer);
      glBufferData(GL_ARRAY_BUFFER, indices * 3 * sizeof(indices), &shapeNormalVector[0], GL_STATIC_DRAW);

      const unsigned int bbIndices = bbIndexVector.size();
      const unsigned int bbLineIndices = bbLinesIndexVector.size();
      const unsigned int bbVertices = bbVertexVector.size();

      glBindVertexArray(0);

      GLuint bbVao;
      glGenVertexArrays(1, &bbVao);
      glBindVertexArray(bbVao);

      GLuint bbIndexBuffer;
      glGenBuffers(1, &bbIndexBuffer);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bbIndexBuffer);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, bbLineIndices * sizeof(uint32_t), &bbLinesIndexVector[0], GL_STATIC_DRAW);

      GLuint bbVertexBuffer;
      glGenBuffers(1, &bbVertexBuffer);
      glBindBuffer(GL_ARRAY_BUFFER, bbVertexBuffer);
      glBufferData(GL_ARRAY_BUFFER, bbVertices * sizeof(uint32_t), &bbVertexVector[0], GL_STATIC_DRAW);
      glVertexAttribPointer(
            0,        // attribute 0 - must match layout in shader
            3,        // size
            GL_FLOAT, // type
            GL_FALSE, // normalized?
            0,  // stride
            (void*)0 // array buffer offset
      );
      glEnableVertexAttribArray(0);

      glBindVertexArray(0);

      // Ensure we can capture the escape key being pressed below
      glfwSetInputMode(window, GLFW_STICKY_KEYS, (int)GL_TRUE);

      GLuint programID = LoadShaders("../src/gl/vert.glsl", "../src/gl/frag.glsl");

      // projection matrix - 45deg fov, 4:3 ratio, display range - 0.1 <-> 100 units
      fov = scene->Camera->Settings.FieldOfView;
      projectionMatrix = glm::perspective(glm::radians(fov), (float)width / (float)height, 0.1f, 100.0f);

      // camera matrix
//   glm::mat4 previousView = glm::lookAt(
//         glm::vec3(0, 0, 35),
//         glm::vec3(0, 0, -1),
//         glm::vec3(0, 1, 0)
//         );

      //viewMatrix = MakeMat4(scene->Camera->CameraToWorld);

      eye = glm::vec3(scene->Camera->eye.x, scene->Camera->eye.y, scene->Camera->eye.z);
      lookAt = glm::vec3(scene->Camera->lookAt.x, scene->Camera->lookAt.y, -scene->Camera->lookAt.z);
      up = glm::vec3(scene->Camera->up.x, scene->Camera->up.y, scene->Camera->up.z);

      // camera matrix

      // model - identity, since model will be at the origin for now
      glm::mat4 model = glm::mat4(1.0f);

      // get uniform handle
      GLuint matrixID = glGetUniformLocation(programID, "mvp");
      GLuint colorInId = glGetUniformLocation(programID, "colorIn");

      glEnable(GL_DEPTH_TEST);
      glDepthFunc(GL_LESS);

      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      //glEnable( GL_PROGRAM_POINT_SIZE );

      //glEnable(GL_CULL_FACE);
      glDepthFunc(GL_LESS);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      glEnable( GL_MULTISAMPLE );

      glClearColor(0.15f, 0.15f, 0.15f, 0.0f);

      glUseProgram(programID);

      do {
         // clear the screen
         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

         // calculate new mvp
         projectionMatrix = glm::perspective(glm::radians(fov), (float)width / (float)height, 0.1f, 100.0f);

         viewMatrix = glm::lookAt(eye, lookAt, up);
         glm::mat4 mvp = projectionMatrix * viewMatrix * model;
         // send transformation matrix to the currently bound shader, in the "mvp" uniform
         glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

         // bounding boxes

         glBindVertexArray(bbVao);

         glEnable(GL_DEPTH_TEST);
         glEnable(GL_BLEND);

         glUniform4f(colorInId, 1.0f, 1.0f, 1.0f, 0.5f);
         //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         glDrawElements(GL_LINES, bbLinesIndexVector.size(), GL_UNSIGNED_INT, (void*)0);

         glDisable(GL_DEPTH_TEST);
         glEnable(GL_BLEND);

         // shapes

         glBindVertexArray(shapeVao);

         glDisable(GL_DEPTH_TEST);
         glDisable(GL_BLEND);

         glUniform4f(colorInId, 0.4f, 0.5f, 0.5f, 0.25f);
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         glDrawElements(GL_TRIANGLES, shapeIndexVector.size(), GL_UNSIGNED_INT, (void*)0);

         glDisable(GL_DEPTH_TEST);
         glEnable(GL_BLEND);

         glUniform4f(colorInId, 1.0f, 1.0f, 1.0f, 0.0625f);
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         glDrawElements(GL_TRIANGLES, shapeIndexVector.size(), GL_UNSIGNED_INT, (void*)0);

         // swap bufffers
         glfwSwapBuffers(window);
         glfwPollEvents();

      } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

      glfwDestroyWindow(window);

      glfwTerminate();
      //exit(EXIT_SUCCESS);
   }
}