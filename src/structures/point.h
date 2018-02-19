//
// Created by Daniel Thompson on 2/18/18.
//

#ifndef POLYTOPE_POINT_H
#define POLYTOPE_POINT_H


class Point {
public:
   Point(float x, float y, float z);
   Point(const Point &p);

   bool operator==(const Point &rhs) const;
   bool operator!=(const Point &rhs) const;

   float Dot(const Point &p) const;

   float x, y, z;


};


#endif //POLYTOPE_POINT_H
