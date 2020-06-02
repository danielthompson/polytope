//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_CONSTANTS_H
#define POLYTOPE_CONSTANTS_H

#include <cmath>
#include <limits>
#include <random>

#include "structures/Vectors.h"
#include "structures/Vectors.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

#ifndef M_SQRT3
#define M_SQRT3 1.73205080757
#endif

namespace Polytope {

   constexpr float Epsilon = .00001f;
   constexpr float HalfEpsilon = Epsilon * 0.5f;
   constexpr float TwoEpsilon = Epsilon * 2.0f;
   constexpr float OffsetEpsilon = 0.002f;
   constexpr float PI = float(M_PI);
   constexpr float OneOverPi = 1.0f / PI;
   constexpr float PIOver180 = PI / 180.0f;
   constexpr float TwoPI = 2.0f * PI;

   constexpr float OneThird = 1.0f / 3.0f;

   constexpr float NOHIT = -1.0f;

   constexpr float PIOver2 = PI / 2.0f;
   constexpr float PIOver3 = PI / 3.0f;
   constexpr float PIOver4 = PI / 4.0f;
   constexpr float PIOver6 = PI / 6.0f;
   constexpr float PIOver12 = PI / 12.0f;

   constexpr float OneOver255 = 1.0f / 255.0f;

   constexpr float Root2 = float(M_SQRT2);
   constexpr float Root3 = float(M_SQRT3);

   constexpr float Infinity = std::numeric_limits<float>::infinity();
   constexpr float DenormMin = std::numeric_limits<float>::denorm_min();

   constexpr float FloatMin = std::numeric_limits<float>::min();
   constexpr float FloatMax = std::numeric_limits<float>::max();
   constexpr unsigned int UnsignedIntMax = std::numeric_limits<unsigned int>::max();
   
   constexpr float MachineEpsilon = std::numeric_limits<float>::epsilon() * 0.5f;

   inline bool WithinEpsilon(float number, float target, float epsilon) {

      if (number > target) {
         return (2 * epsilon + target >= number);
      }
      else
         return (2 * epsilon + number >= target);
   }

   inline bool WithinEpsilon(float number, float target) {

      if (number > target) {
         return (TwoEpsilon + target >= number);
      }
      else
         return (TwoEpsilon + number >= target);
   }

   inline float RadiansBetween(const Vector &v, const Normal &n) {
      return float(std::abs(acos(v.Dot(n))));
   }

   static thread_local std::random_device random_device;

   static thread_local std::mt19937 Generator(random_device());

   inline int RandomUniformBetween(const int floor, const int ceiling) {
      std::uniform_int_distribution<int> distribution(floor, ceiling);
      return distribution(Generator);
   }

   inline unsigned int RandomUniformBetween(const unsigned int floor, const unsigned int ceiling) {
      std::uniform_int_distribution<unsigned int> distribution(floor, ceiling);
      return distribution(Generator);
   }
   
   inline float NormalizedUniformRandom() {
      //static thread_local std::mt19937 Generator;
      std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
      return distribution(Generator);
   }

   inline Vector CosineSampleHemisphere(float u0, float u1) {
      const float r = std::sqrt(u0);
      const float theta = TwoPI * u1;

      const float x = r * std::cos(theta);
      const float y = std::sqrt(std::max(0.0f, 1.0f - u0));
      const float z = r * std::sin(theta);

      return Vector(x, y, z);
   }

   inline float SignedDistanceFromPlane(const Point &pointOnPlane, const Normal &normal, const Point &p) {
      return (p - pointOnPlane).Dot(normal);
   }

   // source - http://www.pbr-book.org/3ed-2018/Shapes/Managing_Rounding_Error.html#x1-ErrorPropagation
   inline float Gamma(const int n) {
      return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
   }
}

#endif //POLYTOPE_CONSTANTS_H