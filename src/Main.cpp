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

void glfw();

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

static const struct
{
   float x, y;
   float r, g, b;
} vertices[3] =
      {
            { -0.6f, -0.4f, 1.f, 0.f, 0.f },
            {  0.6f, -0.4f, 0.f, 1.f, 0.f },
            {   0.f,  0.6f, 0.f, 0.f, 1.f }
      };

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

void glfw(Polytope::AbstractScene* scene)
{
   GLFWwindow* window;
   GLuint vertex_buffer, vertex_shader, fragment_shader, program;
   GLint mvp_location, vpos_location, vcol_location;

   glfwSetErrorCallback(error_callback);

   if (!glfwInit())
      exit(EXIT_FAILURE);

   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

   window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
   if (!window)
   {
      glfwTerminate();
      exit(EXIT_FAILURE);
   }

   glfwSetKeyCallback(window, key_callback);

   glfwMakeContextCurrent(window);
   glbinding::initialize(glfwGetProcAddress);
   glfwSwapInterval(1);


   // returns 1 buffer object names in vertex_buffer
   glGenBuffers(1, &vertex_buffer);

   // bind vertex_buffer to target GL_ARRAY_BUFFER (Vertex attributes)
   glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);

   
   glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

   vertex_shader = glCreateShader(GL_VERTEX_SHADER);
   glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
   glCompileShader(vertex_shader);

   fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
   glCompileShader(fragment_shader);

   program = glCreateProgram();
   glAttachShader(program, vertex_shader);
   glAttachShader(program, fragment_shader);
   glLinkProgram(program);

   mvp_location = glGetUniformLocation(program, "MVP");
   vpos_location = glGetAttribLocation(program, "vPos");
   vcol_location = glGetAttribLocation(program, "vCol");

   glEnableVertexAttribArray(vpos_location);
   glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
                         sizeof(vertices[0]), (void*) 0);
   glEnableVertexAttribArray(vcol_location);
   glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE,
                         sizeof(vertices[0]), (void*) (sizeof(float) * 2));

   while (!glfwWindowShouldClose(window))
   {
      float ratio;
      int width, height;
      mat4x4 m, p, mvp;

      glfwGetFramebufferSize(window, &width, &height);
      ratio = width / (float) height;

      glViewport(0, 0, width, height);
      glClear(GL_COLOR_BUFFER_BIT);

      mat4x4_identity(m);
      mat4x4_rotate_Z(m, m, (float) glfwGetTime());
      mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
      mat4x4_mul(mvp, p, m);

      glUseProgram(program);
      glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat*) mvp);
      glDrawArrays(GL_TRIANGLES, 0, 3);

      glfwSwapBuffers(window);
      glfwPollEvents();
   }

   glfwDestroyWindow(window);

   glfwTerminate();
   exit(EXIT_SUCCESS);
}
