typedef float<3> float3;

export struct soa_hit_result {
   float t;
   unsigned int index;
   bool hits;
};
 
const uniform float float_max = 3.40282346638528859812e+38F;
const uniform unsigned int unsigned_int_max = 4294967294;

inline float3 cross(float3 v0, float3 v1) {
   float<3> retval = {
      v0.y * v1.z - v0.z * v1.y,
      v0.z * v1.x - v0.x * v1.z,
      v0.x * v1.y - v0.y * v1.x
   };

   return retval;
}

inline float dot(float3 v0, float3 v1) {
   return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

export void soa_linear_intersect (
      uniform const float vx[],
      uniform const float vy[],
      uniform const float vz[],
      uniform float rox,
      uniform float roy,
      uniform float roz,
      uniform float rdx,
      uniform float rdy,
      uniform float rdz,
      uniform float &return_t,
      uniform unsigned int &return_face_index,
      uniform bool &return_hits,
      uniform unsigned int num_faces,
      uniform const int pixel_x,
      uniform const int pixel_y,
      uniform const int bounce) {

   uniform const float3 ro = { rox, roy, roz };
   uniform const float3 rd = { rdx, rdy, rdz };

   uniform const unsigned int v1_index_offset = num_faces;
   uniform const unsigned int v2_index_offset = num_faces * 2;

   varying soa_hit_result partial_result;
   partial_result.t = float_max;
   partial_result.index = 0;
   partial_result.hits = false;
   
   foreach (index = 0 ... num_faces) {

      //      t[index] = float_max;

      //      const int xindex = index;
      //      const int yindex = index + yOffset;
      //      const int zindex = index + zOffset;

      //const unsigned int v0index = fv0[index];

      bool debug = false;
      if (pixel_x == 245 && pixel_y == 64 && bounce == 2 && index == 6175) {
         debug = false;
//         print("foo");
      }
      
      const unsigned int v1index = index + v1_index_offset;
      const unsigned int v2index = index + v2_index_offset;
      
      const float3 v0 = { vx[index], vy[index], vz[index] };
      const float3 v1 = { vx[v1index], vy[v1index], vz[v1index] };
      const float3 v2 = { vx[v2index], vy[v2index], vz[v2index] };

      //print("v0: % % %\n", v0x, v0y, v0z);

      // const poly::Vector edge0 = vertex1 - vertex0;
      const float3 e0 = v1 - v0;

      // const poly::Vector edge1 = vertex2 - vertex1;
      const float3 e1 = v2 - v1;

      //poly::Vector planeNormal = edge0.Cross(edge1);
      float3 pn = cross(e0, e1);

      //planeNormal.Normalize();
      const float oneOverLength = 1.0f / sqrt(dot(pn, pn));
      pn *= oneOverLength;

      //const float divisor = planeNormal.Dot(ray.Direction);
      const float divisor = dot(pn, rd);

      if (divisor == 0.0f) {
         // parallel
         continue;
      }

      // const float t = planeNormal.Dot(vertex0 - ray.Origin) / divisor;
      const float ft = (dot(pn, v0 - ro)) / divisor;

      if (ft <= 0 || ft >= partial_result.t) {
         continue;
      }
      // t[index] = ft;

      // const poly::Point hitPoint = ray.GetPointAtT(t);
      const float3 hp = ro + rd * ft;
      const float3 e2 = v0 - v2;

      const float3 p0 = hp - v0;
      const float3 cross0 = cross(e0, p0);
      const float normal0 = dot(cross0, pn);
      const bool pos0 = normal0 > 0;

      if (!pos0)
         continue;
      
      const float3 p1 = hp - v1;
      const float3 cross1 = cross(e1, p1);
      const float normal1 = dot(cross1, pn);
      const bool pos1 = normal1 > 0;

      if (!pos1)
         continue;
      
      const float3 p2 = hp - v2;
      const float3 cross2 = cross(e2, p2);
      const float normal2 = dot(cross2, pn);
      const bool pos2 = normal2 > 0;

      if (!pos2)
         continue;
      
      partial_result.t = ft;
      partial_result.index = index;
      partial_result.hits = true;
   }
   
   for (uniform int i = 0; i < programCount; i++) {
      uniform const float partial_t = extract(partial_result.t, i);
      uniform const unsigned int partial_index = extract(partial_result.index, i);
      uniform const bool less_than = partial_t < return_t;
      return_t = less_than ? partial_t : return_t;
      return_face_index = less_than ? partial_index : return_face_index;
      return_hits = less_than || return_hits;
   }
}
