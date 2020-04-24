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

export void simple(
      uniform float hitx,
      uniform float hity,
      uniform float hitz,
      uniform float pnx,
      uniform float pny,
      uniform float pnz,
      uniform float vertex[],
      uniform float edge[],
      uniform bool out[],
      uniform int count) {
   foreach (index = 0 ... count) {

      const int xindex = index;
      const int yindex = index + 3;
      const int zindex = index + 6;

      // const Polytope::Vector p2 = hitPoint - vertex2;
      const float px = hitx - vertex[xindex];
      const float py = hity - vertex[yindex];
      const float pz = hitz - vertex[zindex]; 

      // const Polytope::Vector cross2 = edge2.Cross(p2);
      const float crossx = edge[yindex] * pz - edge[zindex] * py;
      const float crossy = edge[zindex] * px - edge[xindex] * pz;
      const float crossz = edge[xindex] * py - edge[yindex] * px;

      //const float normal2 = cross2.Dot(planeNormal);
      const float n = crossx * pnx + crossy * pny + crossz * pnz;

      // const bool pos2 = normal2 > 0;
      out[index] = n > 0;
   }
}

export uniform soa_hit_result soa_hits(
      uniform const float vx[],
      uniform const float vy[],
      uniform const float vz[],
      uniform const unsigned int fv0[],
      uniform const unsigned int fv1[],
      uniform const unsigned int fv2[],
      uniform float rox,
      uniform float roy,
      uniform float roz,
      uniform float rdx,
      uniform float rdy,
      uniform float rdz,
      uniform unsigned int count) {

   uniform const float3 ro = { rox, roy, roz };
   uniform const float3 rd = { rdx, rdy, rdz };
   
   varying float t = float_max;
   varying unsigned int original_index = unsigned_int_max;
   
   foreach (index = 0 ... count) { 
      
//      const int xindex = index;
//      const int yindex = index + yOffset;
//      const int zindex = index + zOffset;

      // TODO - gathers required here, maybe pre-pack these per BVH node at file read time?
      const unsigned int v0index = fv0[index];
      float3 v0 = { vx[v0index], vy[v0index], vz[v0index] };
      
      const unsigned int v1index = fv1[index];
      float3 v1 = { vx[v1index], vy[v1index], vz[v1index] };

      const unsigned int v2index = fv2[index];
      float3 v2 = { vx[v2index], vy[v2index], vz[v2index] };

      //print("v0: % % %\n", v0x, v0y, v0z);

      // const Polytope::Vector edge0 = vertex1 - vertex0;
      const float3 e0 = v1 - v0;

      // const Polytope::Vector edge1 = vertex2 - vertex1;
      const float3 e1 = v2 - v1;

      //Polytope::Vector planeNormal = edge0.Cross(edge1);
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
      const float new_t = (dot(pn, v0 - ro)) / divisor;
      
      if (new_t < t && new_t > 0) {
         // const Polytope::Point hitPoint = ray.GetPointAtT(t);
         const float3 hp = ro + rd * new_t;

         //print("hp: % % %\n", hpx, hpy, hpz);

         // const Polytope::Vector edge2 = vertex0 - vertex2;
         const float3 e2 = v0 - v2;

         const float3 p0 = hp - v0;
         const float3 cross0 = cross(e0, p0);
         const float normal0 = dot(cross0, pn);
         const bool pos0 = normal0 > 0;

         const float3 p1 = hp - v1;
         const float3 cross1 = cross(e1, p1);
         const float normal1 = dot(cross1, pn);
         const bool pos1 = normal1 > 0;

         const float3 p2 = hp - v2;
         const float3 cross2 = cross(e2, p2);
         const float normal2 = dot(cross2, pn);
         const bool pos2 = normal2 > 0;

         //print("inside: % % %\n", pos0, pos1, pos2);

         const bool inside = pos0 && pos1 && pos2;
         if (inside) {
            t = new_t;
            original_index = programIndex;
         }
      }
   }

   uniform float min_t = float_max;
   uniform unsigned int min_t_index = unsigned_int_max;
   
   for (uniform int i = 0; i < programCount; i++) {
      uniform float possible_t = extract(t, i);
      if (possible_t < min_t) {
         min_t = possible_t;
         min_t_index = extract(original_index, i);
      }
   }

   print("t: %\n", t);
   print("original index: %\n", original_index);
   print("min_t: %\n", min_t);
   print("min_t_index: %\n", min_t_index);
   
   uniform soa_hit_result result;
   result.index = min_t_index;
   result.t = min_t;
   
   return result;
}

export uniform soa_hit_result soa_intersect(
      uniform const float vx[],
      uniform const float vy[],
      uniform const float vz[],
      uniform const unsigned int fv0[],
      uniform const unsigned int fv1[],
      uniform const unsigned int fv2[],
      uniform float rox,
      uniform float roy,
      uniform float roz,
      uniform float rdx,
      uniform float rdy,
      uniform float rdz,
      uniform float t[],
      uniform unsigned int count) {

   uniform const float3 ro = { rox, roy, roz };
   uniform const float3 rd = { rdx, rdy, rdz };
   
   varying soa_hit_result partial_result;
   partial_result.t = float_max;
   partial_result.index = 0;
   
   foreach (index = 0 ... count) {
      
      //t[index] = float_max;
      
      //      const int xindex = index;
      //      const int yindex = index + yOffset;
      //      const int zindex = index + zOffset;
      
      const unsigned int v0index = fv0[index];
      float3 v0 = { vx[v0index], vy[v0index], vz[v0index] };
      
      const unsigned int v1index = fv1[index];
      float3 v1 = { vx[v1index], vy[v1index], vz[v1index] };
      
      const unsigned int v2index = fv2[index];
      float3 v2 = { vx[v2index], vy[v2index], vz[v2index] };
      
      //print("v0: % % %\n", v0x, v0y, v0z);
      
      // const Polytope::Vector edge0 = vertex1 - vertex0;
      const float3 e0 = v1 - v0;
      
      // const Polytope::Vector edge1 = vertex2 - vertex1;
      const float3 e1 = v2 - v1;
      
      //Polytope::Vector planeNormal = edge0.Cross(edge1);
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
      
      if (ft <= 0) {
         continue;
      }
      
//      t[index] = ft;
      
      // const Polytope::Point hitPoint = ray.GetPointAtT(t);
      const float3 hp = ro + rd * ft;
      
      //print("hp: % % %\n", hpx, hpy, hpz);
      
      // const Polytope::Vector edge2 = vertex0 - vertex2;
      const float3 e2 = v0 - v2; 
      
      const float3 p0 = hp - v0;
      const float3 cross0 = cross(e0, p0);
      const float normal0 = dot(cross0, pn);
      const bool pos0 = normal0 > 0;
      
      const float3 p1 = hp - v1;
      const float3 cross1 = cross(e1, p1);
      const float normal1 = dot(cross1, pn);
      const bool pos1 = normal1 > 0;
      
      const float3 p2 = hp - v2;
      const float3 cross2 = cross(e2, p2);
      const float normal2 = dot(cross2, pn);
      const bool pos2 = normal2 > 0;
      
      //print("inside: % % %\n", pos0, pos1, pos2);
      
      const bool inside = pos0 && pos1 && pos2;

      partial_result.t = inside ? ft : partial_result.t;
      partial_result.index = inside ? index : partial_result.index;
      
//      if (!inside)
//         t[index] = float_max;
   }
   
   uniform soa_hit_result uniform_result;
   
   for (uniform int i = 0; i < programCount; i++) {
      uniform float partial_t = extract(partial_result.t, i);
      if (partial_t < uniform_result.t) {
         uniform_result.t = partial_t;
         uniform_result.index = extract(partial_result.index, i);
      }
   }
   
   return uniform_result;
   
}

export void soa_intersect_expanded(
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
      //uniform float t[],
      uniform unsigned int num_faces) {

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

      // TODO - gathers required here, maybe pre-pack these per BVH node at file read time?
      //const unsigned int v0index = fv0[index];
      
      const unsigned int v1index = index + v1_index_offset;
      const unsigned int v2index = index + v2_index_offset;
      
      const float3 v0 = { vx[index], vy[index], vz[index] };
      const float3 v1 = { vx[v1index], vy[v1index], vz[v1index] };
      const float3 v2 = { vx[v2index], vy[v2index], vz[v2index] };

      //print("v0: % % %\n", v0x, v0y, v0z);

      // const Polytope::Vector edge0 = vertex1 - vertex0;
      const float3 e0 = v1 - v0;

      // const Polytope::Vector edge1 = vertex2 - vertex1;
      const float3 e1 = v2 - v1;

      //Polytope::Vector planeNormal = edge0.Cross(edge1);
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

      // const Polytope::Point hitPoint = ray.GetPointAtT(t);
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

export void soa2( 
      uniform float vertex[],
      uniform float rayox,
      uniform float rayoy,
      uniform float rayoz,
      uniform float raydirx,
      uniform float raydiry,
      uniform float raydirz,
      uniform float t[],
      uniform bool inside[],
      uniform unsigned int yOffset,
      uniform unsigned int zOffset,
      uniform int count) {
   
   //print("test");
   //print("test");
   foreach (index = 0 ... count) {

      const int xindex = index;
      const int yindex = index + yOffset;
      const int zindex = index + zOffset;

      const float v0x = vertex[xindex];
      const float v0y = vertex[yindex];
      const float v0z = vertex[zindex];

      const float v1x = vertex[xindex + 1];
      const float v1y = vertex[yindex + 1];
      const float v1z = vertex[zindex + 1];

      const float v2x = vertex[xindex + 2];
      const float v2y = vertex[yindex + 2];
      const float v2z = vertex[zindex + 2];

      //print("v0: % % %\n", v0x, v0y, v0z);
      
      // const Polytope::Vector edge0 = vertex1 - vertex0;
      const float e0x = v1x - v0x;
      const float e0y = v1y - v0y;
      const float e0z = v1z - v0z;

      // const Polytope::Vector edge1 = vertex2 - vertex1;
      const float e1x = v2x - v1x;
      const float e1y = v2y - v1y;
      const float e1z = v2z - v1z;

      //Polytope::Vector planeNormal = edge0.Cross(edge1);
      float pnx = e0y * e1z - e0z * e1y;
      float pny = e0z * e1x - e0x * e1z;
      float pnz = e0x * e1y - e0y * e1x;

      //planeNormal.Normalize();
      const float oneOverLength = 1.0f / sqrt(pnx * pnx + pny * pny + pnz * pnz);
      pnx *= oneOverLength;
      pny *= oneOverLength;
      pnz *= oneOverLength;

      //const float divisor = planeNormal.Dot(ray.Direction);
      const float divisor = pnx * raydirx + pny * raydiry + pnz * raydirz;

      if (divisor == 0.0f) {
         // parallel
         continue;
      }

      // const float t = planeNormal.Dot(vertex0 - ray.Origin) / divisor;
      const float ft = (pnx * (v0x - rayox) + pny * (v0y - rayoy) + pnz * (v0z - rayoz)) / divisor;

      t[index] = ft;

      if (ft <= 0)
         continue;

      // const Polytope::Point hitPoint = ray.GetPointAtT(t);
      const float hpx = rayox + raydirx * ft;
      const float hpy = rayoy + raydiry * ft;
      const float hpz = rayoz + raydirz * ft;

      // const Polytope::Vector edge2 = vertex0 - vertex2;
      const float e2x = v0x - v2x;
      const float e2y = v0y - v2y;
      const float e2z = v0z - v2z;

      // const Polytope::Vector p0 = hitPoint - vertex0;
      const float p0x = hpx - v0x;
      const float p0y = hpy - v0y;
      const float p0z = hpz - v0z;

      // const Polytope::Vector cross0 = edge0.Cross(p0);
      const float cross0x = e0y * p0z - e0z * p0y;
      const float cross0y = e0z * p0x - e0x * p0z;
      const float cross0z = e0x * p0y - e0y * p0x;

      // const float normal0 = cross0.Dot(planeNormal);
      const float normal0 = cross0x * pnx + cross0y * pny + cross0z * pnz;

      // const bool pos0 = normal0 > 0;
      const bool pos0 = normal0 > 0;

//      const Polytope::Vector p1 = hitPoint - vertex1;
//      const Polytope::Vector cross1 = edge1.Cross(p1);
//      const float normal1 = cross1.Dot(planeNormal);
//      const bool pos1 = normal1 > 0;

      const float p1x = hpx - v1x;
      const float p1y = hpy - v1y;
      const float p1z = hpz - v1z;
      const float cross1x = e1y * p1z - e1z * p1y;
      const float cross1y = e1z * p1x - e1x * p1z;
      const float cross1z = e1x * p1y - e1y * p1x;
      const float normal1 = cross1x * pnx + cross1y * pny + cross1z * pnz;
      const bool pos1 = normal1 > 0;

//      const Polytope::Vector p2 = hitPoint - vertex2;
//      const Polytope::Vector cross2 = edge2.Cross(p2);
//      const float normal2 = cross2.Dot(planeNormal);
//      const bool pos2 = normal2 > 0;

      const float p2x = hpx - v2x;
      const float p2y = hpy - v2y;
      const float p2z = hpz - v2z;
      const float cross2x = e2y * p2z - e2z * p2y;
      const float cross2y = e2z * p2x - e2x * p2z;
      const float cross2z = e2x * p2y - e2y * p2x;
      const float normal2 = cross2x * pnx + cross2y * pny + cross2z * pnz;
      const bool pos2 = normal2 > 0;

      bool inside = pos0 && pos1 && pos2;
      if (!inside)
         t[index] = -1;
   }
}

