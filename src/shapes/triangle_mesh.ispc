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
