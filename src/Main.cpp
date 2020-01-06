//
// Created by Daniel Thompson on 2/18/18.
//

#include <iostream>
#include <sstream>
#include <map>

#define GLFW_INCLUDE_NONE
#include <glbinding/gl/gl.h>
#include <glbinding/glbinding.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../lib/linmath.h"

#include "utilities/OptionsParser.h"
#include "utilities/Common.h"
#include "scenes/SceneBuilder.h"
#include "integrators/PathTraceIntegrator.h"
#include "samplers/HaltonSampler.h"
#include "filters/BoxFilter.h"
#include "runners/TileRunner.h"
#include "films/PNGFilm.h"
#include "parsers/PBRTFileParser.h"
#include "shapes/TriangleMesh.h"

#ifdef __CYGWIN__
#include "platforms/win32-cygwin.h"
#endif

Polytope::Logger Log;

void segfaultHandler(int signalNumber) {
   Log.WithTime("Detected a segfault. Stacktrace to be implemented...");
#ifdef __CYGWIN__
   //printStack();
#endif
   exit(signalNumber);
}

void signalHandler(int signalNumber) {
   std::ostringstream oss;
   oss << "Received interrupt signal " << signalNumber << ", aborting.";
   Log.WithTime(oss.str());
   exit(signalNumber);
}

bool hasAbortedOnce = false;

void userAbortHandler(int signalNumber) {
   if (hasAbortedOnce) {
      Log.WithTime("Aborting at user request.");
      exit(signalNumber);
   }
   else {
      Log.WithTime("Detected Ctrl-C keypress. Ignoring since it's the first time. Press Ctrl-C again to really quit.");
      hasAbortedOnce = true;
   }
}

void glfw(Polytope::AbstractScene* scene);

int main(int argc, char* argv[]) {

   try {
      Log = Polytope::Logger();

      Polytope::Options options = Polytope::Options();

      if (argc > 0) {
         Polytope::OptionsParser parser(argc, argv);
         options = parser.Parse();
      }

      if (options.help) {
         std::cout << "Polytope by Daniel A. Thompson, built on " << __DATE__ << std::endl;
         fprintf(stderr, R"(
Usage: polytope [options] -inputfile <filename> [-outputfile <filename>]

Rendering options:
   -threads <n>      Number of CPU threads to use for rendering. Optional;
                     defaults to the number of detected logical cores.
   -samples <n>      Number of samples to use per pixel. Optional; overrides
                     the number of samples specified in the scene file.

File options:
   -inputfile        The scene file to render. Currently, PBRT is the only
                     supported file format. Optional but strongly encouraged;
                     defaults to a boring example scene.
   -outputfile       The filename to render to. Currently, PNG is the only
                     supported output file format. Optional; overrides the
                     output filename specified in the scene file, if any;
                     defaults to the input file name (with .png extension).

Other:
   -gl               Render the scene with OpenGL, for reference.
   --help            Print this help text and exit.)");
         std::cout << std::endl;
         exit(0);
      }

      const auto totalRunTimeStart = std::chrono::system_clock::now();

      constexpr unsigned int width = 640;
      constexpr unsigned int height = 480;

      const Polytope::Bounds bounds(width, height);

      const unsigned int concurrentThreadsSupported = std::thread::hardware_concurrency();
      Log.WithTime("Detected " + std::to_string(concurrentThreadsSupported) + " cores.");

      unsigned int usingThreads = concurrentThreadsSupported;

      if (options.threadsSpecified && options.threads > 0 && options.threads <= concurrentThreadsSupported) {
         usingThreads = options.threads;
      }

      Log.WithTime("Using " + std::to_string(usingThreads) + " threads.");

      {
         std::unique_ptr<Polytope::AbstractRunner> runner;
         if (!options.inputSpecified) {
            Log.WithTime("No input file specified, using default scene.");
            Polytope::SceneBuilder sceneBuilder = Polytope::SceneBuilder(bounds);
            Polytope::AbstractScene *scene = sceneBuilder.Default();

            // TODO fix
            // Compile(scene);

            std::unique_ptr<Polytope::AbstractSampler> sampler = std::make_unique<Polytope::HaltonSampler>();
            std::unique_ptr<Polytope::AbstractIntegrator> integrator = std::make_unique<Polytope::PathTraceIntegrator>(scene, 5);

            std::unique_ptr<Polytope::BoxFilter> filter = std::make_unique<Polytope::BoxFilter>(bounds);
            filter->SetSamples(options.samples);
            std::unique_ptr<Polytope::AbstractFilm> film = std::make_unique<Polytope::PNGFilm>(bounds, options.output_filename, std::move(filter));

            runner = std::make_unique<Polytope::TileRunner>(std::move(sampler), scene, std::move(integrator), std::move(film), bounds, options.samples);

         }
         else {
            // load file
            Polytope::PBRTFileParser parser = Polytope::PBRTFileParser();
            runner = parser.ParseFile(options.input_filename);

            // override parsed with options here
            if (options.samplesSpecified) {
               runner->NumSamples = options.samples;
            }
         }

         Log.WithTime(
               std::string("Image is [") +
               std::to_string(runner->Bounds.x) +
               std::string("] x [") +
               std::to_string(runner->Bounds.y) +
               std::string("], ") +
               std::to_string(runner->NumSamples) + " spp.");

         if (options.gl) {
            Log.WithTime("Rasterizing with OpenGL...");
            glfw(runner->Scene);
         }
         else {
            Log.WithTime("Rendering...");

            const auto renderingStart = std::chrono::system_clock::now();

            //   runner->Run();

            std::map<std::thread::id, int> threadMap;
            std::vector<std::thread> threads;
            for (int i = 0; i < usingThreads; i++) {

               Log.WithTime(std::string("Starting thread " + std::to_string(i) + std::string("...")));
               threads.emplace_back(runner->Spawn(i));
               const std::thread::id threadID = threads[i].get_id();
               threadMap[threadID] = i;

            }

            for (int i = 0; i < usingThreads; i++) {
               threads[i].join();
               Log.WithTime(std::string("Joined thread " + std::to_string(i) + std::string(".")));
            }

            const auto renderingEnd = std::chrono::system_clock::now();

            const std::chrono::duration<double> renderingElapsedSeconds = renderingEnd - renderingStart;
            Log.WithTime("Rendering complete in " + std::to_string(renderingElapsedSeconds.count()) + "s.");

            Log.WithTime("Outputting to film...");
            const auto outputStart = std::chrono::system_clock::now();
            runner->Output();
            const auto outputEnd = std::chrono::system_clock::now();

            const std::chrono::duration<double> outputtingElapsedSeconds = outputEnd - outputStart;
            Log.WithTime("Outputting complete in " + std::to_string(outputtingElapsedSeconds.count()) + "s.");
         }


      }

      const auto totalRunTimeEnd = std::chrono::system_clock::now();
      const std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;

      Log.WithTime("Total computation time: " + std::to_string(totalElapsedSeconds.count()) + ".");

      Log.WithTime("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}




static const char* vertex_shader_text =
      "#version 110\n"
      "uniform mat4 MVP;\n"
      "attribute vec3 vCol;\n"
      "attribute vec2 vPos;\n"
      "varying vec3 color;\n"
      "void main()\n"
      "{\n"
      "    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
      "    color = vCol;\n"
      "}\n";

static const char* fragment_shader_text =
      "#version 110\n"
      "varying vec3 color;\n"
      "void main()\n"
      "{\n"
      "    gl_FragColor = vec4(color, 1.0);\n"
      "}\n";

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

static const GLfloat g_vertex_buffer_data[] = {
      -1.0f,-1.0f,-1.0f, // triangle 1 : begin
      -1.0f,-1.0f, 1.0f,
      -1.0f, 1.0f, 1.0f, // triangle 1 : end
      1.0f, 1.0f,-1.0f, // triangle 2 : begin
      -1.0f,-1.0f,-1.0f,
      -1.0f, 1.0f,-1.0f, // triangle 2 : end
      1.0f,-1.0f, 1.0f,
      -1.0f,-1.0f,-1.0f,
      1.0f,-1.0f,-1.0f,
      1.0f, 1.0f,-1.0f,
      1.0f,-1.0f,-1.0f,
      -1.0f,-1.0f,-1.0f,
      -1.0f,-1.0f,-1.0f,
      -1.0f, 1.0f, 1.0f,
      -1.0f, 1.0f,-1.0f,
      1.0f,-1.0f, 1.0f,
      -1.0f,-1.0f, 1.0f,
      -1.0f,-1.0f,-1.0f,
      -1.0f, 1.0f, 1.0f,
      -1.0f,-1.0f, 1.0f,
      1.0f,-1.0f, 1.0f,
      1.0f, 1.0f, 1.0f,
      1.0f,-1.0f,-1.0f,
      1.0f, 1.0f,-1.0f,
      1.0f,-1.0f,-1.0f,
      1.0f, 1.0f, 1.0f,
      1.0f,-1.0f, 1.0f,
      1.0f, 1.0f, 1.0f,
      1.0f, 1.0f,-1.0f,
      -1.0f, 1.0f,-1.0f,
      1.0f, 1.0f, 1.0f,
      -1.0f, 1.0f,-1.0f,
      -1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f,
      -1.0f, 1.0f, 1.0f,
      1.0f,-1.0f, 1.0f
};

// One color for each vertex. They were generated randomly.
static const GLfloat g_color_buffer_data[] = {
      0.583f,  0.771f,  0.014f,
      0.609f,  0.115f,  0.436f,
      0.327f,  0.483f,  0.844f,
      0.822f,  0.569f,  0.201f,
      0.435f,  0.602f,  0.223f,
      0.310f,  0.747f,  0.185f,
      0.597f,  0.770f,  0.761f,
      0.559f,  0.436f,  0.730f,
      0.359f,  0.583f,  0.152f,
      0.483f,  0.596f,  0.789f,
      0.559f,  0.861f,  0.639f,
      0.195f,  0.548f,  0.859f,
      0.014f,  0.184f,  0.576f,
      0.771f,  0.328f,  0.970f,
      0.406f,  0.615f,  0.116f,
      0.676f,  0.977f,  0.133f,
      0.971f,  0.572f,  0.833f,
      0.140f,  0.616f,  0.489f,
      0.997f,  0.513f,  0.064f,
      0.945f,  0.719f,  0.592f,
      0.543f,  0.021f,  0.978f,
      0.279f,  0.317f,  0.505f,
      0.167f,  0.620f,  0.077f,
      0.347f,  0.857f,  0.137f,
      0.055f,  0.953f,  0.042f,
      0.714f,  0.505f,  0.345f,
      0.783f,  0.290f,  0.734f,
      0.722f,  0.645f,  0.174f,
      0.302f,  0.455f,  0.848f,
      0.225f,  0.587f,  0.040f,
      0.517f,  0.713f,  0.338f,
      0.053f,  0.959f,  0.120f,
      0.393f,  0.621f,  0.362f,
      0.673f,  0.211f,  0.457f,
      0.820f,  0.883f,  0.371f,
      0.982f,  0.099f,  0.879f
};



GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path){

   // Create the shaders
   GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
   GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

   // Read the Vertex Shader code from the file
   std::string VertexShaderCode;
   std::ifstream VertexShaderStream(vertex_file_path, std::ios::in);
   if(VertexShaderStream.is_open()){
      std::stringstream sstr;
      sstr << VertexShaderStream.rdbuf();
      VertexShaderCode = sstr.str();
      VertexShaderStream.close();
   }else{
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

void glfw(Polytope::AbstractScene* scene)
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

      window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
      if (!window) {
         fprintf(stderr, "Failed to open GLFW window :/");
         glfwTerminate();
         exit(EXIT_FAILURE);
      }

      glfwSetKeyCallback(window, key_callback);

      glfwMakeContextCurrent(window);
      glbinding::initialize(glfwGetProcAddress);
      glfwSwapInterval(1);

   }
   // end GLFW init

   // actual drawing code

   gl::GLuint VertexArrayID;
   glGenVertexArrays(1, &VertexArrayID);
   glBindVertexArray(VertexArrayID);

   Polytope::TriangleMesh* mesh = (Polytope::TriangleMesh *)(scene->Shapes[0]);
   const unsigned int indices = mesh->Faces.size() * 3;

   std::vector<unsigned int> flatIndexVector(indices, 0);
   std::vector<float> flatVertexVector(indices * 3, 0.f);

   for (const Polytope::Point3ui &face : mesh->Faces) {
      flatIndexVector.push_back(face.x);
      flatIndexVector.push_back(face.y);
      flatIndexVector.push_back(face.z);

      flatVertexVector.push_back(mesh->Vertices[face.x].x);
      flatVertexVector.push_back(mesh->Vertices[face.x].y);
      flatVertexVector.push_back(mesh->Vertices[face.x].z);

      flatVertexVector.push_back(mesh->Vertices[face.y].x);
      flatVertexVector.push_back(mesh->Vertices[face.y].y);
      flatVertexVector.push_back(mesh->Vertices[face.y].z);

      flatVertexVector.push_back(mesh->Vertices[face.z].x);
      flatVertexVector.push_back(mesh->Vertices[face.z].y);
      flatVertexVector.push_back(mesh->Vertices[face.z].z);
   }

   // buffer for vertex indices
   GLuint indexBuffer;
   glGenBuffers(1, &indexBuffer);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
   glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices * sizeof(indices), &flatIndexVector[0], GL_STATIC_DRAW);

   // buffer for vertex locations
   GLuint vertexBuffer;
   glGenBuffers(1, &vertexBuffer);
   glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
   glBufferData(GL_ARRAY_BUFFER, indices * 3 * sizeof(indices), &flatVertexVector[0], GL_STATIC_DRAW);

   // Ensure we can capture the escape key being pressed below
   glfwSetInputMode(window, GLFW_STICKY_KEYS, (int)GL_TRUE);

   GLuint programID = LoadShaders("../src/gl/vert.glsl", "../src/gl/frag.glsl");

   glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

   // projection matrix - 45deg fov, 4:3 ratio, display range - 0.1 <-> 100 units
   glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);

   // camera matrix
   glm::mat4 view = glm::lookAt(
         glm::vec3(4, 3, 3),
         glm::vec3(0, 0, 0),
         glm::vec3(0, 1, 0)
         );

   // model - identity, since model will be at the origin for now
   glm::mat4 model = glm::mat4(1.0f);

   // mvp
   glm::mat4 mvp = projection * view * model;

   // get uniform handle
   GLuint matrixID = glGetUniformLocation(programID, "mvp");

   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);

   do {
      // clear the screen
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      glUseProgram(programID);

      // send transformation matrix to the currently bound shader, in the "mvp" uniform
      glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

      // 1st attribute buffer - the vertices
      glEnableVertexAttribArray(0);
      glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
      glVertexAttribPointer(
            0,        // attribute 0 - must match layout in shader
            3,        // size
            GL_FLOAT, // type
            GL_FALSE, // normalized?
            0,  // stride
            (void*)0 // array buffer offset
            );

      glEnableVertexAttribArray(1);
//      glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
//      glVertexAttribPointer(
//            1,
//            3,
//            GL_FLOAT,
//            GL_FALSE,
//            0,
//            (void*)0
//            );

      // draw the triangle
      // starting from vertex 0; 3 vertices total -> 1 triangle
      glDrawElements(GL_TRIANGLES, flatIndexVector.size(), GL_UNSIGNED_INT, (void*)0);

      glDisableVertexAttribArray(0);

      // swap bufffers
      glfwSwapBuffers(window);
      glfwPollEvents();

   } while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

   glfwDestroyWindow(window);

   glfwTerminate();
   exit(EXIT_SUCCESS);
}
