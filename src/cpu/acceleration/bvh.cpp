//
// Created by daniel on 6/9/20.
//

#include <queue>
#include <algorithm>
#include "bvh.h"

namespace poly {

   struct node_info {
      poly::BoundingBox bb;
      poly::Point centroid;
      std::pair<unsigned int, unsigned int> index;
   };

   unsigned int bvh::bound(const std::vector<poly::Mesh*>& meshes) {
      this->meshes = meshes;
      
      // reserve enough space for all mesh faces
      unsigned int total_faces = 0;
      for (const poly::Mesh* mesh : meshes) {
         total_faces += mesh->num_faces;
      }

      std::vector<std::pair<unsigned int, unsigned int>> all_indices;
      all_indices.reserve(total_faces);

      Point root_min = { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
      Point root_max = { -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};

      std::vector<std::vector<struct node_info>> nodes_info;
      
      // generate index vector per mesh
      for (unsigned int mesh_index = 0; mesh_index < meshes.size(); mesh_index++) {
         const poly::Mesh* mesh = meshes[mesh_index];
         
         // calculate bounding boxes and centroids for each face
         std::vector<struct node_info> node_info;
         node_info.reserve(mesh->num_faces);
         
         for (unsigned int face_index = 0; face_index < mesh->num_faces; face_index++) {
            Point face_min, face_max;

            // get face
            const Point v0 = {
                  mesh->x_packed[mesh->fv0[face_index]],
                  mesh->y_packed[mesh->fv0[face_index]],
                  mesh->z_packed[mesh->fv0[face_index]] };

            const Point v1 = {
                  mesh->x_packed[mesh->fv1[face_index]],
                  mesh->y_packed[mesh->fv1[face_index]],
                  mesh->z_packed[mesh->fv1[face_index]] };

            const Point v2 = {
                  mesh->x_packed[mesh->fv2[face_index]],
                  mesh->y_packed[mesh->fv2[face_index]],
                  mesh->z_packed[mesh->fv2[face_index]] };

            // calculate face's bounding box
            face_min.x = v0.x < v1.x ? v0.x : v1.x;
            face_min.x = face_min.x < v2.x ? face_min.x : v2.x;
            face_min.y = v0.y < v1.y ? v0.y : v1.y;
            face_min.y = face_min.y < v2.y ? face_min.y : v2.y;
            face_min.z = v0.z < v1.z ? v0.z : v1.z;
            face_min.z = face_min.z < v2.z ? face_min.z : v2.z;

            face_max.x = v0.x > v1.x ? v0.x : v1.x;
            face_max.x = face_max.x > v2.x ? face_max.x : v2.x;
            face_max.y = v0.y > v1.y ? v0.y : v1.y;
            face_max.y = face_max.y > v2.y ? face_max.y : v2.y;
            face_max.z = v0.z > v1.z ? v0.z : v1.z;
            face_max.z = face_max.z > v2.z ? face_max.z : v2.z;

            // update root bounding box
            root_min.x = root_min.x < face_min.x ? root_min.x : face_min.x;
            root_min.y = root_min.y < face_min.y ? root_min.y : face_min.y;
            root_min.z = root_min.z < face_min.z ? root_min.z : face_min.z;

            root_max.x = root_max.x > face_max.x ? root_max.x : face_max.x;
            root_max.y = root_max.y > face_max.y ? root_max.y : face_max.y;
            root_max.z = root_max.z > face_max.z ? root_max.z : face_max.z;

            node_info.push_back({
               poly::BoundingBox {face_min, face_max },
               // calculate centroid
               poly::Point {(v0.x + v1.x + v2.x) * OneThird, 
                            (v0.y + v1.y + v2.y) * OneThird, 
                            (v0.z + v1.z + v2.z) * OneThird },
               { mesh_index, face_index}
            });
            
            all_indices.emplace_back(mesh_index, face_index);
         }

         nodes_info.push_back(node_info);
      }
      
      // create root node
      root = new bvh_node();
      root->bb = { root_min, root_max };
      root->indices = all_indices;

      std::queue<std::pair<bvh_node*, unsigned int>> q;
      q.emplace(root, 0);
      num_nodes++;
      
      // subdivide
      while (!q.empty()) {
         auto element = q.front();
         q.pop();
         bvh_node* node = element.first;
         const unsigned int depth = element.second;
         std::vector<std::pair<unsigned int, unsigned int>> indices = node->indices;

         // base case 0
         if (depth > 35)
            continue;
         
         // base case 1
         if (indices.size() < 10)
            continue;
         
         // sort by centroid along each axis
         
         float x_min = std::numeric_limits<float>::infinity();
         float y_min = std::numeric_limits<float>::infinity();
         float z_min = std::numeric_limits<float>::infinity();
         float x_max = -std::numeric_limits<float>::infinity();
         float y_max = -std::numeric_limits<float>::infinity();
         float z_max = -std::numeric_limits<float>::infinity();

         // calculate extents
         for (const std::pair<unsigned int, unsigned int> &index : indices) {
            unsigned int mesh_index = index.first;
            unsigned int face_index = index.second;
            poly::Point centroid = nodes_info[mesh_index][face_index].centroid;
            
            x_min = x_min < centroid.x ? x_min : centroid.x;
            y_min = y_min < centroid.y ? y_min : centroid.y;
            z_min = z_min < centroid.z ? z_min : centroid.z;

            x_max = x_max > centroid.x ? x_max : centroid.x;
            y_max = y_max > centroid.y ? y_max : centroid.y;
            z_max = z_max > centroid.z ? z_max : centroid.z;
         }

         float x_extent = x_max - x_min;
         float y_extent = y_max - y_min;
         float z_extent = z_max - z_min;

         // split on midpoint
         // TODO split on median or an edge

         poly::Axis next_axis = Axis::x;
         float split = (x_max + x_min) * 0.5f;
         
         if (y_extent >= x_extent && y_extent >= z_extent) {
            next_axis = poly::Axis::y;
            split = (y_max + y_min) * 0.5f;
         }
         else if (z_extent >= x_extent && z_extent >= y_extent) {
            next_axis = poly::Axis::z;
            split = (z_max + z_min) * 0.5f;
         }

         // partition & create child node bounding boxes
         std::vector<std::pair<unsigned int, unsigned int>> low_indices, high_indices;

         for (const std::pair<unsigned int, unsigned int> &index : indices) {
            unsigned int mesh_index = index.first;
            unsigned int face_index = index.second;
            poly::Point centroid = nodes_info[mesh_index][face_index].centroid;
            if (centroid[next_axis] < split)
               low_indices.push_back(index);
            else
               high_indices.push_back(index);
         }
         
         // base case 2 - we're not making any progress - turn the current node into a leaf
         if (low_indices.size() == node->indices.size() || high_indices.size() == node->indices.size()) {
            continue;
         }

         // calculate bounding boxes for child nodes
         
         Point high_min = {
               std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity()
         };

         Point high_max = {
               -std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity()
         };

         for (const std::pair<unsigned int, unsigned int> &index : high_indices) {
            unsigned int mesh_index = index.first;
            unsigned int face_index = index.second;
            high_min.x = high_min.x < nodes_info[mesh_index][face_index].bb.p0.x ? high_min.x : nodes_info[mesh_index][face_index].bb.p0.x;
            high_min.y = high_min.y < nodes_info[mesh_index][face_index].bb.p0.y ? high_min.y : nodes_info[mesh_index][face_index].bb.p0.y;
            high_min.z = high_min.z < nodes_info[mesh_index][face_index].bb.p0.z ? high_min.z : nodes_info[mesh_index][face_index].bb.p0.z;

            high_max.x = high_max.x > nodes_info[mesh_index][face_index].bb.p1.x ? high_max.x : nodes_info[mesh_index][face_index].bb.p1.x;
            high_max.y = high_max.y > nodes_info[mesh_index][face_index].bb.p1.y ? high_max.y : nodes_info[mesh_index][face_index].bb.p1.y;
            high_max.z = high_max.z > nodes_info[mesh_index][face_index].bb.p1.z ? high_max.z : nodes_info[mesh_index][face_index].bb.p1.z;
         }
         
         bvh_node* high_child = new bvh_node();
         //high_child->axis = next_axis;
         high_child->indices = high_indices;
         high_child->high = nullptr;
         high_child->low = nullptr;
         high_child->bb = { high_min, high_max};
         node->high = high_child;
         
         q.emplace(high_child, depth + 1);
         num_nodes++;
         
         Point low_min = {
               std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity(),
               std::numeric_limits<float>::infinity()
         };

         Point low_max = {
               -std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity(),
               -std::numeric_limits<float>::infinity()
         };

         for (const std::pair<unsigned int, unsigned int> &index : low_indices) {
            unsigned int mesh_index = index.first;
            unsigned int face_index = index.second;
            low_min.x = low_min.x < nodes_info[mesh_index][face_index].bb.p0.x ? low_min.x : nodes_info[mesh_index][face_index].bb.p0.x;
            low_min.y = low_min.y < nodes_info[mesh_index][face_index].bb.p0.y ? low_min.y : nodes_info[mesh_index][face_index].bb.p0.y;
            low_min.z = low_min.z < nodes_info[mesh_index][face_index].bb.p0.z ? low_min.z : nodes_info[mesh_index][face_index].bb.p0.z;

            low_max.x = low_max.x > nodes_info[mesh_index][face_index].bb.p1.x ? low_max.x : nodes_info[mesh_index][face_index].bb.p1.x;
            low_max.y = low_max.y > nodes_info[mesh_index][face_index].bb.p1.y ? low_max.y : nodes_info[mesh_index][face_index].bb.p1.y;
            low_max.z = low_max.z > nodes_info[mesh_index][face_index].bb.p1.z ? low_max.z : nodes_info[mesh_index][face_index].bb.p1.z;
         }
         
         bvh_node* low_child = new bvh_node();
         //low_child->axis = next_axis;
         low_child->indices = low_indices;
         low_child->low = nullptr;
         low_child->low = nullptr;
         low_child->bb = { low_min, low_max};
         node->low = low_child;
         
         q.emplace(low_child, depth + 1);
         num_nodes++;
         
         node->indices.clear();
      }
      
      this->num_nodes = num_nodes;
      
      return num_nodes;
   }

   bool bvh::hits_compact(const poly::Ray &ray) const {
      const poly::Vector inverse_direction = {
            1.f / ray.Direction.x,
            1.f / ray.Direction.y,
            1.f / ray.Direction.z
      };
      std::queue<compact_bvh_node *> q;
      q.push(compact_root->nodes);
      while (!q.empty()) {
         compact_bvh_node* node = q.front();
         q.pop();
         if (node->bb.Hits(ray, inverse_direction)) {
            // if leaf node
            if (node->indices != nullptr) {
               // intersect faces
               for (unsigned int i = 0; i < node->num_face_indices; i++) {
                  const unsigned int mesh_index = node->indices[i].first;
                  const poly::Mesh* mesh = meshes[mesh_index];
                  const unsigned int face_index = node->indices[i].second;
                  // TODO make this more efficient
                  if (mesh->hits(ray, &face_index, 1)) {
                     // if anything hits, return true
                     return true;
                  }
               }
               // otherwise, keep traversing               
               continue;
            }

            // if interior node
            // TODO push closer child node first (use node's split axis and ray's direction's sign for that axis
            q.push(node + 1);
            q.push(node + node->high_offset);
         }
      }
      // we made it all the way through the tree and nothing hit, so no hit
      return false;
   }
   
   bool bvh::hits(const poly::Ray &ray) const {
      const poly::Vector inverse_direction = {
            1.f / ray.Direction.x,
            1.f / ray.Direction.y,
            1.f / ray.Direction.z
      }; 
      std::queue<bvh_node *> q;
      q.push(root);
      while (!q.empty()) {
         bvh_node* node = q.front();
         q.pop();
         if (node->bb.Hits(ray, inverse_direction)) {
            // if leaf node
            if (node->high == nullptr && node->low == nullptr) {
               // intersect faces
               // TODO make this more efficient
               for (const auto &index : node->indices) {
                  const unsigned int mesh_index = index.first;
                  const poly::Mesh* mesh = meshes[mesh_index];
                  const unsigned int face_index = index.second;
                  
                  if (mesh->hits(ray, &face_index, 1)) {
                     // if anything hits, return true
                     return true;
                  }
               }
               // otherwise, keep traversing               
               continue;
            }
            
            // if interior node
            // TODO push closer child node first (use node's split axis and ray's direction's sign for that axis
            q.push(node->high);
            q.push(node->low);
         }
      }
      // we made it all the way through the tree and nothing hit, so no hit
      return false;
   }

   void bvh::intersect_compact(poly::Ray& ray, poly::Intersection& intersection) const {
      const poly::Vector inverse_direction = {
            1.f / ray.Direction.x,
            1.f / ray.Direction.y,
            1.f / ray.Direction.z
      };
      std::queue<compact_bvh_node *> q;
      q.push(compact_root->nodes);

      while (!q.empty()) {
         compact_bvh_node* node = q.front();
         q.pop();

         // TODO add tmax optimization for BBox hit
         if (node->bb.Hits(ray, inverse_direction)) {
            // if leaf node
            if (node->indices != nullptr) {
               // intersect faces
               for (unsigned int i = 0; i < node->num_face_indices; i++) {
                  const unsigned int mesh_index = node->indices[i].first;
                  poly::Mesh* mesh = meshes[mesh_index];
                  const unsigned int face_index = node->indices[i].second;
                  mesh->intersect(ray, intersection, &face_index, 1);
               }
               // otherwise, keep traversing               
               continue;
            }

            // if interior node
            // TODO push closer child node first (use node's split axis and ray's direction's sign for that axis
            q.push(node + 1);
            q.push(node + node->high_offset);
         }
      }
   }
   
   void bvh::intersect(poly::Ray& ray, poly::Intersection& intersection) const {
      const poly::Vector inverse_direction = {
            1.f / ray.Direction.x,
            1.f / ray.Direction.y,
            1.f / ray.Direction.z
      };
      std::queue<bvh_node *> q;
      q.push(root);
      
      while (!q.empty()) {
         bvh_node* node = q.front();
         q.pop();
            
         // TODO add tmax optimization for BBox hit
         if (node->bb.Hits(ray, inverse_direction)) {
            // if leaf node
            if (node->high == nullptr && node->low == nullptr) {
               // intersect faces
               // TODO make this more efficient
               for (const auto &index : node->indices) {
                  const unsigned int mesh_index = index.first;
                  poly::Mesh* mesh = meshes[mesh_index];
                  const unsigned int face_index = index.second;
                  mesh->intersect(ray, intersection, &face_index, 1);
               }
            }

            // if interior node
            // TODO push closer child node first (use node's split axis and ray's direction's sign for that axis
            q.push(node->high);
            q.push(node->low);
         }
      }
   }
   
   void bvh::metrics() const {
      unsigned int tree_height = 0;
      unsigned int num_interior_nodes = 0;
      unsigned int num_leaf_nodes = 0;
      unsigned int num_single_high_child_nodes = 0;
      unsigned int num_single_low_child_nodes = 0;
      
      std::vector<unsigned int> leaf_counts;
      
      std::queue<std::pair<bvh_node*, unsigned int>> q;
      q.emplace(root, 0);
      while (!q.empty()) {
         const auto element = q.front();
         q.pop();
         bvh_node* node = element.first;
         unsigned int node_height = element.second;
         
         if (tree_height < node_height) {
            tree_height = node_height;
         }
         
         if (node->high == nullptr && node->low == nullptr) {
            num_leaf_nodes++;
            leaf_counts.push_back(node->indices.size());
         }
         else if (node->high == nullptr && node->low != nullptr) {
            num_single_low_child_nodes++;
            q.emplace(node->low, node_height + 1);
         }
         else if (node->high != nullptr && node->low == nullptr) {
            num_single_high_child_nodes++;
            q.emplace(node->high, node_height + 1);
         }
         else if (node->high != nullptr && node->low != nullptr) {
            num_interior_nodes++;
            q.emplace(node->low, node_height + 1);
            q.emplace(node->high, node_height + 1);
         }
      }
      
      std::sort(leaf_counts.begin(), leaf_counts.end());
      
      unsigned int total_faces = 0;
      unsigned int faces_per_leaf_avg = 0;
      unsigned int faces_per_leaf_min = std::numeric_limits<unsigned int>::max();
      unsigned int faces_per_leaf_max = 0;
      
      for (const unsigned int leaf_faces : leaf_counts) {
         total_faces += leaf_faces;
         if (leaf_faces < faces_per_leaf_min) {
            faces_per_leaf_min = leaf_faces;
         }
         if (leaf_faces > faces_per_leaf_max) {
            faces_per_leaf_max = leaf_faces;
         }
      }
      
      faces_per_leaf_avg = total_faces / num_leaf_nodes;
      
      printf("height: %i\n", tree_height);
      printf("leaves: %i\n", num_leaf_nodes);
      printf("good interior: %i\n", num_interior_nodes);
      printf("high interior: %i\n", num_single_high_child_nodes);
      printf("low interior : %i\n", num_single_low_child_nodes);
      printf("total faces : %i\n", total_faces);
      printf("faces per leaf (avg): %i\n", faces_per_leaf_avg);
      printf("faces per leaf (min): %i\n", faces_per_leaf_min);
      printf("faces per leaf (max): %i\n", faces_per_leaf_max);
      
      const float bucket_index_width = (leaf_counts.size()) / 500.0f;
      
//      for (unsigned int i = 0; i < 500; i++) {
//         printf("%i\n", leaf_counts[bucket_index_width * i]);   
//      }
   }

   bvh::~bvh() {
      std::queue<bvh_node*> q;
      q.push(root);
      while (!q.empty()) {
         bvh_node* node = q.front();
         q.pop();
         if (node != nullptr) {
            q.push(node->low);
            q.push(node->high);
            delete node;
         }
      }
      
      delete compact_root;
   }

   unsigned int bvh::compact_helper(bvh_node* node, unsigned int index) {
      if (node != nullptr) {
         int num_child_nodes = 0;
         compact_bvh_node& compact_node = compact_root->nodes[index];
         compact_node.bb = node->bb;

         // leaf
         if (node->low == nullptr) {
            compact_node.indices = &node->indices[0];
            compact_node.num_face_indices = node->indices.size();
         } 
         // interior
         else {
            num_child_nodes = 0;
            num_child_nodes += 1 + compact_helper(node->low, index + 1);
            compact_node.high_offset = num_child_nodes + 1;
            compact_node.indices = nullptr;
            num_child_nodes += 1 + compact_helper(node->high, index + 1 + num_child_nodes);
            
         }

         return num_child_nodes;
      }
   }
   
   void bvh::compact() {
      compact_root = new compact_bvh(num_nodes);
      compact_helper(root, 0);
   }
}
