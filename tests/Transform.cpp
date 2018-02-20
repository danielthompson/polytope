//
// Created by Daniel Thompson on 2/19/18.
//

#include <iostream>
#include "gtest/gtest.h"

#include "../src/structures/Transform.h"

namespace Tests {

    using Polytope::Transform;

    namespace Equality {
        TEST(Transform, Equals) {
           Transform element1 = Transform();
           Transform element2 = Transform();
           EXPECT_EQ(element1, element2);
        }
    }

}