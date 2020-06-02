////
//// Created by daniel on 5/14/20.
////
//
//#ifndef POLYTOPE_GENERATE_RAYS_CUH
//#define POLYTOPE_GENERATE_RAYS_CUH
//
//#include "../../cpu/scenes/AbstractScene.h"
//#include "../gpu_memory_manager.h"
//
//namespace Polytope {
//   class RayGeneratorKernel {
//   public:
//      explicit RayGeneratorKernel(Polytope::AbstractScene *scene, std::shared_ptr<Polytope::GPUMemoryManager> memory_manager);
//      ~RayGeneratorKernel();
//
//      void GenerateRays();
//
//      void CheckRays();
//      
//      Polytope::AbstractScene* scene;
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
//#endif //POLYTOPE_GENERATE_RAYS_CUH
