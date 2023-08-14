//
// Created by Daniel Thompson on 2/19/18.
//

#ifndef POLY_TRANSFORM_H
#define POLY_TRANSFORM_H

#include "Vectors.h"
#include "ray.h"
#include "bounding_box.h"
#include "matrix.h"
//#include "../shapes/abstract_mesh.h"

namespace poly {

   class transform {
   public:

      // constructors

      transform();
      explicit transform(const float values[4][4]);
      explicit transform(float m00, float m01, float m02, float m03,
                         float m10, float m11, float m12, float m13,
                         float m20, float m21, float m22, float m23,
                         float m30, float m31, float m32, float m33);
      explicit transform(const poly::matrix &matrix);
      transform(const poly::matrix &matrix, const poly::matrix &inverse);
      transform(const poly::transform &other) = default;

      // operators

      bool operator==(const poly::transform &rhs) const;
      bool operator!=(const poly::transform &rhs) const;
      poly::transform operator*(const poly::transform &rhs) const;
      poly::transform &operator*=(const poly::transform &rhs);

      // methods

      poly::transform invert() const;

      void apply_in_place(poly::point &p) const;
      poly::point apply(const poly::point &p) const;
      
      void apply_in_place(poly::vector &v) const;
      poly::vector apply(const poly::vector &v) const;

      void apply_in_place(poly::normal &n) const;
      poly::normal Apply(const poly::normal &normal) const;

      void apply_in_place(poly::ray &ray) const;
      poly::ray apply(const poly::ray &ray) const;

      void ApplyInPlace(poly::bounding_box &bb) const;
      poly::bounding_box Apply(const poly::bounding_box &bb) const;

      bool has_scale() const;

      static poly::transform rotate(float angle, float x, float y, float z);

      static poly::transform translate(const poly::vector &delta);
      static poly::transform translate(float x, float y, float z);

      static transform scale(const poly::vector &delta);
      static transform scale(float x, float y, float z);
      static transform scale(float t);

      static transform look_at(const point &eye, const point &look_at, poly::vector &up, bool right_handed = true);

      static transform look_at_left_handed(const poly::point &eye, const poly::point &look_at, poly::vector &up);
      
      // data

      poly::matrix matrix;
      poly::matrix inverse;

      static constexpr poly::vector x_dir = { 1, 0, 0 };
      static constexpr poly::vector y_dir = { 0, 1, 0 };
      static constexpr poly::vector z_dir = { 0, 0, 1 };
   };
}

#endif //POLY_TRANSFORM_H
