//
// Created by Daniel Thompson on 2/24/18.
//

#include "SceneBuilder.h"
#include "../cameras/PerspectiveCamera.h"
#include "NaiveScene.h"
#include "../shading/Material.h"
#include "../shading/brdf/MirrorBRDF.h"

namespace Polytope {

   AbstractScene* SceneBuilder::Default(float x, float y) {
      CameraSettings settings = CameraSettings(x, y, 50.0f);

      Transform identity = Transform();

      std::shared_ptr<AbstractCamera> camera = std::make_shared<PerspectiveCamera>(settings, identity);

      AbstractScene *scene = new NaiveScene(camera);

      // Orange ball

      Material material = Material();

      material.BRDF = MirrorBRDF();
      material.ReflectanceSpectrum = new ReflectanceSpectrum(Solarized.cyan);

      inputTransforms = new Transform[4];
      inputTransforms[0] = Transform.Translate(new Vector(100.0f, 0.0f, -250f));
      inputTransforms[1] = Transform.RotateX(45);
      //inputTransforms[2] = Transform.RotateY(45);
      inputTransforms[2] = Transform.Scale(100f, 100f, 100f);
      inputTransforms[3] = Transform.Translate(new Vector(-.5f, -.5f, -.5f));

      compositeTransforms = Transform.composite(inputTransforms);

      Box box = new Box(compositeTransforms, material);

      scene.addShape(box);

      // yellow ball

      material = new Material();
      material.BRDF = LambertianBRDF;
      material.ReflectanceSpectrum = new ReflectanceSpectrum(Firenze.Beige);

      inputTransforms = new Transform[2];
      //inputTransforms[0] = Transform.Translate(new Vector(-150.0f, -50.0f, 100.0f));
      //inputTransforms[1] = Transform.Scale(55f, 55f, 55f);
      inputTransforms[0] = Transform.Translate(new Vector(-100f, 0f, -200.0f));
      inputTransforms[1] = Transform.Scale(50f);

      compositeTransforms = Transform.composite(inputTransforms);

      Sphere sphere2 = new Sphere(compositeTransforms, material);

      scene.addShape(sphere2);

      // white light

      SpectralPowerDistribution lightSPD = new SpectralPowerDistribution(Color.white, 1000000.0f);

      inputTransforms = new Transform[2];
      inputTransforms[0] = Transform.Translate(new Vector(300, 3300, -1500));
      inputTransforms[1] = Transform.Scale(100f, 100f, 100f);

      compositeTransforms = Transform.composite(inputTransforms);

      Sphere sphere = new Sphere(compositeTransforms, null);

      AbstractLight light = new SphereLight(sphere, lightSPD);

      // skybox

      scene.SkyBoxImage = Skyboxes.Load(Skyboxes.Desert2Captions);

      return scene;
   }

}