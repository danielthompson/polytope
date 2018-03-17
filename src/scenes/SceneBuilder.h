//
// Created by Daniel Thompson on 2/24/18.
//

#ifndef POLYTOPE_SCENEBUILDER_H
#define POLYTOPE_SCENEBUILDER_H

#include "AbstractScene.h"
#include "../shading/ReflectanceSpectrum.h"

namespace Polytope {

   class SceneBuilder {
   public:
      SceneBuilder(Polytope::Bounds bounds) : _bounds(bounds) { }
      AbstractScene *Default();

   private:
      Polytope::Bounds _bounds;

      ReflectanceSpectrum FirenzeGreen = ReflectanceSpectrum(70, 137, 102);
      ReflectanceSpectrum FirenzeBeige = ReflectanceSpectrum(255, 240, 165);
      ReflectanceSpectrum FirenzeYellow = ReflectanceSpectrum(255, 176, 59);
      ReflectanceSpectrum FirenzeOrange = ReflectanceSpectrum(182, 73, 38);
      ReflectanceSpectrum FirenzeRed = ReflectanceSpectrum(142, 40, 0);

      ReflectanceSpectrum SolarizedBase03 = ReflectanceSpectrum(0, 43, 54);
      ReflectanceSpectrum SolarizedBase02 = ReflectanceSpectrum(7, 54, 66);
      ReflectanceSpectrum SolarizedBase01 = ReflectanceSpectrum(88, 110, 117);
      ReflectanceSpectrum SolarizedBase00 = ReflectanceSpectrum(101, 123, 131);
      ReflectanceSpectrum SolarizedBase0 = ReflectanceSpectrum(131, 148, 150);
      ReflectanceSpectrum SolarizedBase1 = ReflectanceSpectrum(147, 161, 161);
      ReflectanceSpectrum SolarizedBase2 = ReflectanceSpectrum(238, 232, 213);
      ReflectanceSpectrum SolarizedBase3 = ReflectanceSpectrum(253, 246, 227);
      ReflectanceSpectrum Solarizedyellow = ReflectanceSpectrum(181, 137, 0);
      ReflectanceSpectrum Solarizedorange = ReflectanceSpectrum(203, 75, 22);
      ReflectanceSpectrum Solarizedred = ReflectanceSpectrum(220, 50, 47);
      ReflectanceSpectrum Solarizedmagenta = ReflectanceSpectrum(211, 54, 130);
      ReflectanceSpectrum Solarizedviolet = ReflectanceSpectrum(108, 113, 196);
      ReflectanceSpectrum Solarizedblue = ReflectanceSpectrum(38, 139, 210);
      ReflectanceSpectrum Solarizedcyan = ReflectanceSpectrum(42, 161, 152);
      ReflectanceSpectrum Solarizedgreen = ReflectanceSpectrum(133, 153, 0);
   };
}

#endif //POLYTOPE_SCENEBUILDER_H
