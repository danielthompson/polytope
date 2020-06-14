//
// Created by daniel on 5/17/20.
//

#ifndef POLY_COMMON_DEVICE_FUNCTIONS_CUH
#define POLY_COMMON_DEVICE_FUNCTIONS_CUH

namespace poly {


   
   __device__ inline void normalize(float3 &v) {
      const float one_over_length = 1.f / norm3df(v.x, v.y, v.z);
//      const float one_over_length = 1.f / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
      v.x *= one_over_length;
      v.y *= one_over_length;
      v.z *= one_over_length;
   }

   __device__ inline float3 normalize(const float3 &v) {
//      const float one_over_length = 1.f / sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
      const float one_over_length = 1.f / norm3df(v.x, v.y, v.z);;
      return {
            v.x * one_over_length,
            v.y * one_over_length,
            v.z * one_over_length
      };
   }

   __device__ inline float3 matrix_apply_point(const float* d_matrix, const float3 d_p) {
      const float w = d_p.x * d_matrix[12] + d_p.y * d_matrix[13] + d_p.z * d_matrix[14] + d_matrix[15];
      const float divisor = 1.f / w;

      return {
            (d_p.x * d_matrix[0] + d_p.y * d_matrix[1] + d_p.z * d_matrix[2] + d_matrix[3]) * divisor,
            (d_p.x * d_matrix[4] + d_p.y * d_matrix[5] + d_p.z * d_matrix[6] + d_matrix[7]) * divisor,
            (d_p.x * d_matrix[8] + d_p.y * d_matrix[9] + d_p.z * d_matrix[10] + d_matrix[11]) * divisor
      };
   }

   __device__ inline float3 matrix_apply_vector(const float* d_matrix, const float3 d_v) {
      return {
            d_v.x * d_matrix[0] + d_v.y * d_matrix[1] + d_v.z * d_matrix[2],
            d_v.x * d_matrix[4] + d_v.y * d_matrix[5] + d_v.z * d_matrix[6],
            d_v.x * d_matrix[8] + d_v.y * d_matrix[9] + d_v.z * d_matrix[10]
      };
   }

   __device__ inline float3 operator-(const float3 &a, const float3 &b) {
      return {a.x - b.x, a.y - b.y, a.z - b.z};
   }

   __device__ inline float3 operator+(const float3 &a, const float3 &b) {
      return {a.x + b.x, a.y + b.y, a.z + b.z};
   }

   __device__ inline float3 operator*(const float3 &a, const float t) {
      return {a.x * t, a.y * t, a.z * t};
   }
   
   __device__ inline float3 fma3(const float3 &a, const float3 &b, const float3 &c) {
      return {fma(a.x, b.x, c.x), fma(a.y, b.y, c.y), fma(a.z, b.z, c.z)};
   }

   __device__ inline float3 operator*(const float3 &a, const float3 b) {
      return {a.x * b.x, a.y * b.y, a.z * b.z};
   }

   __device__ inline void operator*=(float3 &a, const float t) {
      a.x *= t;
      a.y *= t;
      a.z *= t;
   }

   __device__ inline void operator+=(float3 &a, const float3 &b) {
      a.x += b.x;
      a.y += b.y;
      a.z += b.z;
   }

   __device__ inline float3 cross(const float3 &a, const float3 &b) {
      return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
      };
   }

   __device__ inline float dot(const float3 &a, const float3 &b) {
      return a.x * b.x + a.y * b.y + a.z * b.z;
   }
   
   __device__ inline float3 cosine_sample_hemisphere(const float u0, const float u1) {
      const float r = sqrtf(u0);
      const float theta = M_PI * 2 * u1;

      const float x = r * cosf(theta);
      const float y = sqrtf(max(0.0f, 1.0f - u0));
      const float z = r * sinf(theta);

      return make_float3(x, y, z);
   } 

}

#endif //POLY_COMMON_DEVICE_FUNCTIONS_CUH
