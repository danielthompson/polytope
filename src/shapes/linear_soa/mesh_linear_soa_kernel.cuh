//
// Created by daniel on 5/12/20.
//

#ifndef POLYTOPE_MESH_LINEAR_SOA_KERNEL_CU_H
#define POLYTOPE_MESH_LINEAR_SOA_KERNEL_CU_H

void initialize_unpacked_mesh(const float *h_x, const float *h_y, const float *h_z, const int num_verts);
void free_mesh();
void linear_intersect(Polytope::Ray &ray);

#endif //POLYTOPE_MESH_LINEAR_SOA_KERNEL_CU_H
