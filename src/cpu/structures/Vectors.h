//
// Created by Daniel on 17-Dec-19.
//

#ifndef POLY_VECTORS_H
#define POLY_VECTORS_H

#include <functional>

namespace poly {
   enum class axis {
      x = 0,
      y,
      z
   };

   class normal {
   public:
      float x, y, z;

      normal() : x(0), y(0), z(0) { };
      normal(const float x, const float y, const float z) : x(x), y(y), z(z) { }
      normal(const normal &n) = default;

      bool operator==(const normal &rhs) const;
      normal operator*(const float t) const;

      void flip();
      void normalize();
   };

   class vector {
   public:
      float x, y, z;

      vector() : x(0), y(0), z(0) { };
      vector(const float x, const float y, const float z);
      vector(const vector &v) = default;
      explicit vector(const poly::normal &n);

      bool operator==(const vector &rhs) const;
      bool operator!=(const vector &rhs) const;
      float operator[] (int index) const;
      vector operator*(const float t) const;
      vector operator-(const vector &rhs) const;
      vector operator+(const vector &rhs) const;
      vector operator-() const;

      float dot(const vector &v) const;
      float dot(const poly::normal &n) const;
      float length() const;
      float length_squared() const;
      void normalize();
      vector cross(const vector &rhs) const;
      vector cross(const poly::normal &rhs) const;
   };

   class point {
   public:
      float x, y, z;

      point() : x(0), y(0), z(0) { };
      point(const float x, const float y, const float z)  : x(x), y(y), z(z) {}
      point(const point &p) : x(p.x), y(p.y), z(p.z) { }

      bool operator<(const point &rhs) const;
      bool operator==(const point &rhs) const;
      bool operator!=(const point &rhs) const;
      poly::vector operator-(const point &rhs) const;
      point operator+(const point &rhs) const;
      point operator+(const poly::vector &rhs) const;
      float &operator[] (const int index);
      float &operator[] (const poly::axis axis);
      point &operator+=(const poly::vector &rhs);
      point &operator+=(const poly::normal &rhs);

      float dot(const point &p) const;
      float dot(const vector &v) const;
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
   struct hash<poly::point>
   {
      std::size_t operator()(const poly::point& k) const
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

#endif //POLY_VECTORS_H
