//
// Created by Daniel Thompson on 2/24/18.
//

#include "SceneBuilder.h"
#include "../cameras/PerspectiveCamera.h"
#include "NaiveScene.h"
#include "../shading/Material.h"
#include "../shading/brdf/MirrorBRDF.h"
#include "../shapes/Sphere.h"
#include "../shading/SpectralPowerDistribution.h"
#include "../lights/AbstractLight.h"
#include "../lights/PointLight.h"
#include "../lights/SphereLight.h"

namespace Polytope {

   AbstractScene* SceneBuilder::Default() {
      CameraSettings settings = CameraSettings(_bounds, 50);

      Transform identity = Transform();

      std::shared_ptr<AbstractCamera> camera = std::make_shared<PerspectiveCamera>(settings, identity);

      AbstractScene *scene = new NaiveScene(camera);

      // Orange ball

      std::shared_ptr<Material> material = std::make_shared<Material>(

         std::make_unique<MirrorBRDF>(),
         Solarizedcyan

      );

      Transform objectToWorld =
            Transform::Translate(100, 0, -200)
            * Transform::Scale(50);

      scene->Shapes.push_back(std::make_shared<Sphere>(objectToWorld, material));

      // yellow ball


      objectToWorld =
            Transform::Translate(-100, 0, -200)
            * Transform::Scale(50);

      scene->Shapes.push_back(std::make_shared<Sphere>(objectToWorld, material));

      // white light

      SpectralPowerDistribution lightSPD = SpectralPowerDistribution(255, 0, 0);

//      inputTransforms = new Transform[2];
//      inputTransforms[0] = Transform.Translate(new Vector(300, 3300, -1500));
//      inputTransforms[1] = Transform.Scale(100f, 100f, 100f);
//
//      compositeTransforms = Transform.composite(inputTransforms);
//
//      Sphere sphere = new Sphere(compositeTransforms, null);
//
//      AbstractLight light = new SphereLight(sphere, lightSPD);

      //scene->Lights.push_back(std::make_shared<PointLight>(lightSPD, Point(0, 1000, 500)));

      material = std::make_shared<Material>(std::make_unique<MirrorBRDF>(), FirenzeBeige);

      objectToWorld =
         Transform::Translate(0, 0, -300)
         * Transform::Scale(25);

      std::shared_ptr<Sphere> sphere = std::make_shared<Sphere>(objectToWorld, material);

      std::shared_ptr<SphereLight> sphereLight = std::make_shared<SphereLight>(lightSPD, sphere);

      sphere->Light = sphereLight;

      scene->Lights.push_back(sphereLight);

      scene->Shapes.push_back(sphere);

      // skybox

      //scene.SkyBoxImage = Skyboxes.Load(Skyboxes.Desert2Captions);

      return scene;
   }

}