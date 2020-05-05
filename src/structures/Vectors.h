//
// Created by Daniel on 17-Dec-19.
//

#ifndef POLYTOPE_VECTORS_H
#define POLYTOPE_VECTORS_H

#include <functional>

namespace Polytope {
   enum class Axis {
      x = 0,
      y,
      z
   };

   class Normal {
   public:
      float x, y, z;

      Normal() : x(0), y(0), z(0) { };
      Normal(const float x, const float y, const float z) : x(x), y(y), z(z) { }
      Normal(const Normal &n) = default;

      bool operator==(const Normal &rhs) const;
      Normal operator*(const float t) const;

      void Flip();
      void Normalize();
   };

   class Vector {
   public:
      float x, y, z;

      Vector() : x(0), y(0), z(0) { };
      Vector(const float x, const float y, const float z);
      Vector(const Vector &v) = default;
      explicit Vector(const Normal &n);

      bool operator==(const Vector &rhs) const;
      bool operator!=(const Vector &rhs) const;
      float operator[] (int index) const;
      Vector operator*(const float t) const;
      Vector operator-(const Vector &rhs) const;
      Vector operator+(const Vector &rhs) const;
      Vector operator-() const;

      float Dot(const Vector &v) const;
      float Dot(const Normal &n) const;
      float Length() const;
      float LengthSquared() const;
      void Normalize();
      Vector Cross(const Vector &rhs) const;
      Vector Cross(const Normal &rhs) const;
   };

   class Point {
   public:
      float x, y, z;

      Point() : x(0), y(0), z(0) { };
      Point(const float x, const float y, const float z)  : x(x), y(y), z(z) {}
      Point(const Point &p) : x(p.x), y(p.y), z(p.z) { }

      bool operator<(const Point &rhs) const;
      bool operator==(const Point &rhs) const;
      bool operator!=(const Point &rhs) const;
      Vector operator-(const Point &rhs) const;
      Point operator+(const Point &rhs) const;
      Point operator+(const Vector &rhs) const;
      float operator[] (const int index) const;
      float operator[] (const Axis axis) const;
      Point &operator+=(const Vector &rhs);
      Point &operator+=(const Normal &rhs);

      float Dot(const Point &p) const;
      float Dot(const Vector &v) const;
   };

   class Point3i {
   public:
      int x, y, z;
      Point3i() : x(0), y(0), z(0) { };
      Point3i(const int x, const int y, const int z) : x(x), y(y), z(z) { };
   };

   class Point3ui {
   public:
      unsigned int x, y, z;
      Point3ui() : x(0), y(0), z(0) { };
      Point3ui(const unsigned int x, const unsigned int y, const unsigned int z) : x(x), y(y), z(z) { };
   };

}

namespace std {
   template <>
   struct hash<Polytope::Point>
   {
      std::size_t operator()(const Polytope::Point& k) const
      {
         // Compute individual hash values for first, second and third
         // http://stackoverflow.com/a/1646913/126995
         std::size_t res = 17;
         res = res * 31 + hash<float>()( k.x );
         res = res * 31 + hash<float>()( k.y );
         res = res * 31 + hash<float>()( k.z );
         return res;
      }
   };
}

#endif //POLYTOPE_VECTORS_H
