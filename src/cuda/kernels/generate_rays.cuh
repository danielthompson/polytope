////
//// Created by daniel on 5/14/20.
////
//
//#ifndef POLY_GENERATE_RAYS_CUH
//#define POLY_GENERATE_RAYS_CUH
//
//#include "../../cpu/scenes/Scene.h"
//#include "../gpu_memory_manager.h"
//
//namespace poly {
//   class RayGeneratorKernel {
//   public:
//      explicit RayGeneratorKernel(poly::Scene *scene, std::shared_ptr<poly::GPUMemoryManager> memory_manager);
//      ~RayGeneratorKernel();
//
//      void GenerateRays();
//
//      void CheckRays();
//      
//      poly::Scene* scene;
//      std::shared_ptr<GPUMemoryManager> memory_manager;
//
//      struct params {
//         unsigned int width;
//         unsigned int height;
//         float fov;
//         struct DeviceCamera* camera;
//      };
//   };
//}
//
//
//#endif //POLY_GENERATE_RAYS_CUH
