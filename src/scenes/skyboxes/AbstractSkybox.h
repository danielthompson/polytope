//
// Created by Daniel Thompson on 2019-06-26.
//

#ifndef POLYTOPE_ABSTRACTSKYBOX_H
#define POLYTOPE_ABSTRACTSKYBOX_H

#include "../../shading/SpectralPowerDistribution.h"

class AbstractSkybox {

public:
   virtual Polytope::SpectralPowerDistribution GetSpd(Polytope::Vector &v) const;
};


#endif //POLYTOPE_ABSTRACTSKYBOX_H
