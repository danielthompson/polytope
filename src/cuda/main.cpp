//
// Created by daniel on 5/13/20.
//

#include <iostream>
#include <chrono>
#include <thread>
#include "../common/utilities/Common.h"
#include "../common/utilities/OptionsParser.h"
#include "../common/structures/Point2.h"
#include "../common/parsers/PBRTFileParser.h"
#include "kernels/generate_rays.cuh"
#include "../cpu/shapes/linear_soa/mesh_linear_soa.h"
#include "gpu_memory_manager.h"
#include "kernels/path_tracer.cuh"
#include "png_output.h"

Polytope::Logger Log;

void cuda_run(Polytope::AbstractScene* scene);

int main(int argc, char* argv[]) {
   try {
      Log = Polytope::Logger();

      Polytope::Options options = Polytope::Options();

      if (argc > 0) {
         Polytope::OptionsParser parser(argc, argv);
         options = parser.Parse();
      }

      if (options.help) {
         std::cout << "Polytope (CUDA version) by Daniel A. Thompson, built on " << __DATE__ << std::endl;
         fprintf(stderr, R"(
Usage: polytope_cuda [options] -inputfile <filename> [-outputfile <filename>]

Rendering options:
   -samples <n>      Number of samples to use per pixel. Optional; overrides
                     the number of samples specified in the scene file.

File options:
   -inputfile        The scene file to render. Currently, PBRT is the only
                     supported file format. Required.
   -outputfile       The filename to render to. Currently, PNG is the only
                     supported output file format. Optional; overrides the
                     output filename specified in the scene file, if any;
                     defaults to the input file name (with .png extension).

Other:
   --help            Print this help text and exit.)");
         std::cout << std::endl;
         exit(0);
      }

      const auto totalRunTimeStart = std::chrono::system_clock::now();

      constexpr unsigned int width = 640;
      constexpr unsigned int height = 480;

      const Polytope::Bounds bounds(width, height);

      Log.WithTime("Using 1 CPU thread.");

      {
         if (!options.inputSpecified) {
            Log.WithTime("No input file specified. Quitting.");
            exit(1);
         }
         
         // load file
         Polytope::PBRTFileParser parser = Polytope::PBRTFileParser();
         const auto runner = parser.ParseFile(options.input_filename);
         
         // override parsed with options here
         if (options.samplesSpecified) {
            runner->NumSamples = options.samples;
         }
         
         Log.WithTime(
               std::string("Image is [") +
               std::to_string(runner->Bounds.x) +
               std::string("] x [") +
               std::to_string(runner->Bounds.y) +
               std::string("], ") +
               std::to_string(runner->NumSamples) + " spp.");


         Log.WithTime("Rendering...");

         const auto renderingStart = std::chrono::system_clock::now();
         cuda_run(runner->Scene);
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

      const auto totalRunTimeEnd = std::chrono::system_clock::now();
      const std::chrono::duration<double> totalElapsedSeconds = totalRunTimeEnd - totalRunTimeStart;

      Log.WithTime("Total computation time: " + std::to_string(totalElapsedSeconds.count()) + ".");

      Log.WithTime("Exiting Polytope.");
   }
   catch (const std::exception&) {
      return EXIT_FAILURE;
   }
}

void cuda_run(Polytope::AbstractScene* scene) {
   // put pixels 
   // determine ray sample positions
   // generate rays

   std::shared_ptr<Polytope::GPUMemoryManager> memory_manager = std::make_shared<Polytope::GPUMemoryManager>();
   memory_manager->MallocSamples();
   
   Polytope::RayGeneratorKernel ray_kernel(scene, memory_manager);
   ray_kernel.GenerateRays();
   ray_kernel.CheckRays();
   
   // push geometry
   for (const auto mesh : scene->Shapes) {
      auto* cast_mesh = dynamic_cast<Polytope::MeshLinearSOA*>(mesh);
      memory_manager->AddMesh(cast_mesh);
   }
   
   // walk rays through the scene
   Polytope::PathTracerKernel path_tracer_kernel(memory_manager);
   path_tracer_kernel.Trace();
   
   // output to file
   Polytope::OutputPNG output;
   output.Output(memory_manager);
}