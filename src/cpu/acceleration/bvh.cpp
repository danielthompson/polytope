//
// Created by daniel on 6/9/20.
//

#include <queue>
#include <algorithm>
#include <stack>
#include <cassert>
#include "bvh.h"

namespace poly {

   struct triangle_info {
      poly::BoundingBox bb;
      poly::Point centroid;
      std::pair<unsigned int, unsigned int> index;
   };

   struct bucket_info {
      poly::BoundingBox low_bb;
      poly::BoundingBox high_bb;
      std::vector<std::pair<unsigned int, unsigned int>> indices;
      unsigned int num_indices;
      float low_surface_area_ratio;
      float high_surface_area_ratio;
   };
   
   unsigned int bvh::bound(const std::vector<poly::Mesh*>& meshes) {
      this->meshes = meshes;
      
      // reserve enough space for all mesh faces
      unsigned int total_faces = 0;
      for (const poly::Mesh* mesh : meshes) {
         total_faces += mesh->num_faces;
      }

      master_indices.reserve(total_faces);

      Point root_min = { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
      Point root_max = { -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};

      std::vector<std::vector<struct triangle_info>> triangles_info;
      
      // generate index vector per mesh
      for (unsigned int mesh_index = 0; mesh_index < meshes.size(); mesh_index++) {
         const poly::Mesh* mesh = meshes[mesh_index];
         
         // calculate bounding boxes and centroids for each face
         std::vector<struct triangle_info> triangle_info;
         triangle_info.reserve(mesh->num_faces);
         
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

            triangle_info.push_back({
               poly::BoundingBox { face_min, face_max },
               // calculate centroid
               poly::Point {(v0.x + v1.x + v2.x) * OneThird, 
                            (v0.y + v1.y + v2.y) * OneThird, 
                            (v0.z + v1.z + v2.z) * OneThird },
               { mesh_index, face_index}
            });

            master_indices.emplace_back(mesh_index, face_index);
         }

         triangles_info.push_back(triangle_info);
      }
      
      // create root node
      root = new bvh_node();
      root->bb = { root_min, root_max };
      root->indices = master_indices;
      
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
         
         // base case 1
         if (indices.size() == 1)
            continue;

         // base case 1.5
         const float node_bb_sa = node->bb.surface_area();
         if (node_bb_sa == 0.f) {
            // this node is too small to split
            continue;
         }
         
         // get centroid extremes by axis
         
         float x_centroid_min = std::numeric_limits<float>::infinity();
         float y_centroid_min = std::numeric_limits<float>::infinity();
         float z_centroid_min = std::numeric_limits<float>::infinity();
         float x_centroid_max = -std::numeric_limits<float>::infinity();
         float y_centroid_max = -std::numeric_limits<float>::infinity();
         float z_centroid_max = -std::numeric_limits<float>::infinity();

         // calculate extents
         for (const std::pair<unsigned int, unsigned int> &index : indices) {
            unsigned int mesh_index = index.first;
            unsigned int face_index = index.second;
            poly::Point centroid = triangles_info[mesh_index][face_index].centroid;

            x_centroid_min = x_centroid_min < centroid.x ? x_centroid_min : centroid.x;
            y_centroid_min = y_centroid_min < centroid.y ? y_centroid_min : centroid.y;
            z_centroid_min = z_centroid_min < centroid.z ? z_centroid_min : centroid.z;

            x_centroid_max = x_centroid_max > centroid.x ? x_centroid_max : centroid.x;
            y_centroid_max = y_centroid_max > centroid.y ? y_centroid_max : centroid.y;
            z_centroid_max = z_centroid_max > centroid.z ? z_centroid_max : centroid.z;
         }

         float x_centroid_midpoint = (x_centroid_max + x_centroid_min) * 0.5f;
         float y_centroid_midpoint = (y_centroid_max + y_centroid_min) * 0.5f;
         float z_centroid_midpoint = (z_centroid_max + z_centroid_min) * 0.5f;
         
         const float x_centroid_extent = x_centroid_max - x_centroid_min;
         const float y_centroid_extent = y_centroid_max - y_centroid_min;
         const float z_centroid_extent = z_centroid_max - z_centroid_min;
         
         // base case: all triangles in this node have the same centroid. nothing to do but make it a leaf and move on
         if (x_centroid_extent == y_centroid_extent
            && y_centroid_extent == z_centroid_extent
            && z_centroid_extent == 0.f) 
            continue;

         poly::Axis split_axis;
         float chosen_axis_centroid_midpoint;
         float chosen_axis_centroid_extent;
         float chosen_axis_centroid_min;
         
         // we choose to split on the axis that has the greatest extent between its centroids
         if (x_centroid_extent >= y_centroid_extent && x_centroid_extent >= z_centroid_extent) {
            split_axis = poly::Axis::x;
            chosen_axis_centroid_midpoint = x_centroid_midpoint;
            chosen_axis_centroid_extent = x_centroid_extent;
            chosen_axis_centroid_min = x_centroid_min;
         }
         else if (y_centroid_extent >= x_centroid_extent && y_centroid_extent >= z_centroid_extent) {
            split_axis = poly::Axis::y;
            chosen_axis_centroid_midpoint = y_centroid_midpoint;
            chosen_axis_centroid_extent = y_centroid_extent;
            chosen_axis_centroid_min = y_centroid_min;
         }
         else {
            split_axis = poly::Axis::z;
            chosen_axis_centroid_midpoint = z_centroid_midpoint;
            chosen_axis_centroid_extent = z_centroid_extent;
            chosen_axis_centroid_min = z_centroid_min;
         }

         // now, we create up to 10 buckets and partition the node's triangles into them
         const unsigned int num_buckets = 10 < node->indices.size() ? 10 : node->indices.size();
         const unsigned int num_split_positions = num_buckets - 1;
         const float num_split_positions_inverse = 1.f / (float)num_split_positions;
         
         const float bucket_width = chosen_axis_centroid_extent / (float)num_buckets;
         const float bucket_width_inverse = 1.f / bucket_width;
         
         // partition & create child node bounding boxes
         std::vector<bucket_info> buckets;
         buckets.reserve(num_buckets);
         
         // initialize buckets
         for (int bucket_index = 0; bucket_index < num_buckets; bucket_index++) {
            buckets.push_back({ });
         }
         
         // assign each triangle in this node to a bucket based on its centroid
         for (const std::pair<unsigned int, unsigned int> &index : indices) {
            unsigned int mesh_index = index.first;
            unsigned int face_index = index.second;
            poly::Point triangle_centroid = triangles_info[mesh_index][face_index].centroid;
            const float triangle_centroid_in_split_axis = triangle_centroid[split_axis];
            // determine which bucket this triangle goes into
            const float raw_bucket_index = (triangle_centroid_in_split_axis - chosen_axis_centroid_min) / (float)bucket_width;
            unsigned int bucket_index = (unsigned int)raw_bucket_index;
            if (bucket_index == num_buckets)
               bucket_index--;
            buckets[bucket_index].indices.push_back(index);
            buckets[bucket_index].num_indices++;
         }
         
         // calculate the SAH cost for just turning this into a leaf node

         
         constexpr float intersection_cost = 1.f;
         constexpr float traversal_cost = 0.0625f;
         const float leaf_sah_cost = node->indices.size() * intersection_cost;
         
         // forward scan to compute BB and SAH cost for the low buckets
         std::vector<float> bucket_forward_sah_cost;
         bucket_forward_sah_cost.reserve(num_buckets);
         float num_low_buckets_sum = 0.f;

         poly::BoundingBox low_bb;

         // initialize bounding box to first face in the first bucket  
         {
            const unsigned int mesh_index = buckets[0].indices[0].first;
            const unsigned int face_index = buckets[0].indices[0].second;
            low_bb = triangles_info[mesh_index][face_index].bb;
         }

         // forward scan for the low buckets
         for (int bucket_index = 0; bucket_index < num_buckets; bucket_index++) {
            struct bucket_info& this_bucket = buckets[bucket_index];
            for (int index_pair_index = 0; index_pair_index < this_bucket.num_indices; index_pair_index++) {
               const unsigned int mesh_index = this_bucket.indices[index_pair_index].first;
               const unsigned int face_index = this_bucket.indices[index_pair_index].second;
               bool debug = false;
               if (depth == 1)
                  debug = true;
               low_bb.UnionInPlace(triangles_info[mesh_index][face_index].bb);
               assert(node->bb.p0.x <= low_bb.p0.x);
               assert(node->bb.p0.y <= low_bb.p0.y);
               assert(node->bb.p0.z <= low_bb.p0.z);
               assert(low_bb.p1.x <= node->bb.p1.x);
               assert(low_bb.p1.y <= node->bb.p1.y);
               assert(low_bb.p1.z <= node->bb.p1.z);
            }
            this_bucket.low_bb = low_bb;
            num_low_buckets_sum += this_bucket.indices.size();
            this_bucket.low_surface_area_ratio = (low_bb.surface_area() / node_bb_sa);
            // a child's bb should never be bigger than the parent's bb
            assert(this_bucket.low_surface_area_ratio >= 0.f);
            assert(this_bucket.low_surface_area_ratio <= 1.f);
            const float scan_cost = (low_bb.surface_area() / node_bb_sa) * (num_low_buckets_sum * intersection_cost);
            bucket_forward_sah_cost.push_back(scan_cost);
         }
         
         poly::BoundingBox high_bb;
         {
            const unsigned int mesh_index = buckets[num_buckets - 1].indices[0].first;
            const unsigned int face_index = buckets[num_buckets - 1].indices[0].second;
            high_bb = triangles_info[mesh_index][face_index].bb;
         }

         std::vector<float> bucket_backward_sah_cost;
         bucket_backward_sah_cost.reserve(num_buckets);
         float num_high_buckets_sum = 0.f;
         
         // backward scan for the high buckets
         for (int bucket_index = num_buckets - 1; bucket_index >= 0; bucket_index--) {
            struct bucket_info& this_bucket = buckets[bucket_index];
            for (const auto& index_pair : this_bucket.indices) {
               const unsigned int mesh_index = index_pair.first;
               const unsigned int face_index = index_pair.second;
               high_bb.UnionInPlace(triangles_info[mesh_index][face_index].bb);
            }
            this_bucket.high_bb = high_bb;
            num_high_buckets_sum += this_bucket.indices.size();
            this_bucket.high_surface_area_ratio = (high_bb.surface_area() / node_bb_sa);
            // a child's bb should never be bigger than the parent's bb
            assert(this_bucket.high_surface_area_ratio >= 0.f);
            assert(this_bucket.high_surface_area_ratio <= 1.f);
            const float scan_cost = (high_bb.surface_area() / node_bb_sa) * (num_high_buckets_sum * intersection_cost);
            bucket_backward_sah_cost.push_back(scan_cost);
         }
         
         //std::reverse(bucket_backward_sah_cost.begin(), bucket_backward_sah_cost.end());
         
         // calculate actual costs
         std::vector<float> total_sah_cost;
         total_sah_cost.reserve(num_buckets - 1);
         for (unsigned int bucket_index = 0; bucket_index < num_buckets - 1; bucket_index++) {
            const float sah = traversal_cost + bucket_forward_sah_cost[bucket_index] + bucket_backward_sah_cost[num_buckets - bucket_index - 1];
            total_sah_cost.push_back(sah);
         };
         
         const float leaf_cost = node->indices.size() * intersection_cost;
         
         // check to see if the leaf cost is less than every SAH cost
         bool should_make_leaf = true;
         for (const auto split_cost : total_sah_cost) {
            if (leaf_cost > split_cost)
               should_make_leaf = false;
         }

         // base case 2
         if (should_make_leaf) {
            continue;
         } 
         
         unsigned int best_split_index = 0;
         float best_split_cost = poly::FloatMax;
         // find lowest cost split
         for (unsigned int bucket_index = 0; bucket_index < num_buckets - 1; bucket_index++) {
            if (total_sah_cost[bucket_index] < best_split_cost) {
               best_split_cost = total_sah_cost[bucket_index];
               best_split_index = bucket_index;
            }
         }

         std::vector<std::pair<unsigned int, unsigned int>> low_indices;
         for (unsigned int bucket_index = 0; bucket_index <= best_split_index; bucket_index++) {
            low_indices.insert(low_indices.end(), buckets[bucket_index].indices.begin(), buckets[bucket_index].indices.end());
         }
         
         std::vector<std::pair<unsigned int, unsigned int>> high_indices;
         for (unsigned int bucket_index = best_split_index + 1; bucket_index < num_buckets; bucket_index++) {
            high_indices.insert(high_indices.end(), buckets[bucket_index].indices.begin(), buckets[bucket_index].indices.end());
         }

         node->axis = split_axis;
         
         assert(low_indices.size() + high_indices.size() == indices.size());
         
         // base case 3 - we're not making any progress - turn the current node into a leaf
         if (low_indices.size() == node->indices.size() || high_indices.size() == node->indices.size()) {
            continue;
         }

         bvh_node* high_child = new bvh_node();
         high_child->indices = high_indices;
         high_child->high = nullptr;
         high_child->low = nullptr;
         // TODO - this is the centroid BB but should be the triangle BB
         high_child->bb = buckets[best_split_index].high_bb;
         node->high = high_child;
         
         q.emplace(high_child, depth + 1);
         num_nodes++;
         
         bvh_node* low_child = new bvh_node();
         low_child->indices = low_indices;
         low_child->high = nullptr;
         low_child->low = nullptr;
         low_child->bb = buckets[best_split_index].low_bb;
         node->low = low_child;
         
         q.emplace(low_child, depth + 1);
         num_nodes++;
         
         node->indices.clear();
      }
      
      return num_nodes;
   }

   bool bvh::hits_compact(const poly::Ray &ray) const {
      const poly::Vector inverse_direction = {
            1.f / ray.Direction.x,
            1.f / ray.Direction.y,
            1.f / ray.Direction.z
      };
      std::stack<compact_bvh_node *> stack;
      stack.push(compact_root->nodes);
      while (!stack.empty()) {
         compact_bvh_node* node = stack.top();
         stack.pop();
         if (node->bb.Hits(ray, inverse_direction)) {
            // if leaf node
            if (node->is_leaf()) {
               // intersect faces
               for (unsigned int i = 0; i < node->num_faces; i++) {
                  // TODO make this more efficient
                  const unsigned int mesh_index = compact_root->leaf_ordered_indices[node->face_index_offset + i].first;
                  const poly::Mesh* mesh = meshes[mesh_index];
                  const unsigned int face_index = compact_root->leaf_ordered_indices[node->face_index_offset + i].second;
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
            stack.push(node + 1);
            stack.push(node + node->high_child_offset);
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
      std::stack<bvh_node *> stack;
      stack.push(root);
      while (!stack.empty()) {
         bvh_node* node = stack.top();
         stack.pop();
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
            stack.push(node->high);
            stack.push(node->low);
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
      
      bool neg_dir[3] = { inverse_direction.x < 0, inverse_direction.y < 0, inverse_direction.z < 0 };
      
      std::stack<compact_bvh_node *> stack;
      stack.push(compact_root->nodes);

      while (!stack.empty()) {
         compact_bvh_node* node = stack.top();
         stack.pop();

         if (node->bb.Hits(ray, inverse_direction)) {
            // if leaf node
            if (node->is_leaf()) {
               // intersect faces
               for (unsigned int i = 0; i < node->num_faces; i++) {
                  const unsigned int mesh_index = compact_root->leaf_ordered_indices[node->face_index_offset + i].first;
                  poly::Mesh* mesh = meshes[mesh_index];
                  const unsigned int face_index = compact_root->leaf_ordered_indices[node->face_index_offset + i].second;
                  mesh->intersect(ray, intersection, &face_index, 1);
               }
               // otherwise, keep traversing               
               continue;
            }

            // if interior node
            // push closer child node second (use node's split axis and ray's direction's sign for that axis
            if (neg_dir[(int)node->get_axis()]) {
               stack.push(node + 1);
               stack.push(node + node->high_child_offset);
            }
            else {
               stack.push(node + node->high_child_offset);
               stack.push(node + 1);
            }
         }
      }
   }
   
   void bvh::intersect(poly::Ray& ray, poly::Intersection& intersection) const {
      const poly::Vector inverse_direction = {
            1.f / ray.Direction.x,
            1.f / ray.Direction.y,
            1.f / ray.Direction.z
      };
      std::stack<bvh_node *> stack;
      stack.push(root);
      
      while (!stack.empty()) {
         bvh_node* node = stack.top();
         stack.pop();
            
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
            stack.push(node->high);
            stack.push(node->low);
         }
      }
   }
   
   void bvh::metrics() const {
      unsigned int tree_height = 0;
      unsigned int num_interior_nodes = 0;
      unsigned int num_leaf_nodes = 0;
      unsigned int num_single_high_child_nodes = 0;
      unsigned int num_single_low_child_nodes = 0;
      
      float leaf_ratio_sum = 0.f;
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
         
         // leaf
         if (node->high == nullptr && node->low == nullptr) {
            num_leaf_nodes++;
            leaf_counts.push_back(node->indices.size());
            
            const float bb_sa = node->bb.surface_area();
            float t_sa = 0.0f;
            for (const auto &indices : node->indices) {
               const poly::Mesh* mesh = meshes[indices.first];
               const unsigned int face_index = indices.second;
               t_sa += mesh->surface_area(face_index);
            }
            
            const float ratio = bb_sa / t_sa;
            leaf_ratio_sum += ratio;
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
      
      const float leaf_ratio_avg = leaf_ratio_sum / (float)num_leaf_nodes;
      
      std::sort(leaf_counts.begin(), leaf_counts.end());
      
      unsigned int total_faces = 0;
      float faces_per_leaf_avg = 0;
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
      
      faces_per_leaf_avg = (float)total_faces / (float)num_leaf_nodes;
      
      printf("height: %i\n", tree_height);
      printf("leaves: %i\n", num_leaf_nodes);
      printf("good interior: %i\n", num_interior_nodes);
      printf("high interior: %i\n", num_single_high_child_nodes);
      printf("low interior : %i\n", num_single_low_child_nodes);
      printf("total faces : %i\n", total_faces);
      printf("faces per leaf (avg): %f\n", faces_per_leaf_avg);
      printf("faces per leaf (min): %i\n", faces_per_leaf_min);
      printf("faces per leaf (max): %i\n", faces_per_leaf_max);
      printf("bb : tri SA ratio (avg): %f\n", leaf_ratio_avg);
      
      const float bucket_index_width = (leaf_counts.size()) / 500.0f;
      
//      for (unsigned int i = 0; i < 500; i++) {
//         printf("%i\n", leaf_counts[bucket_index_width * i]);   
//      }
   }

   unsigned int bvh::compact_helper(bvh_node* node, unsigned int index) {
      if (node != nullptr) {
         unsigned int num_child_nodes = 0;
         compact_bvh_node& compact_node = compact_root->nodes[index];
         compact_node.bb = node->bb;
         compact_node.set_axis(node->axis);

         // leaf
         if (node->low == nullptr) {
            // 1. set leaf's face_index_offset
            compact_node.face_index_offset = compact_root->leaf_ordered_indices.size();
            
            // 2. append original leaf node's indices to leaf_ordered_indices
            for (const auto &indices : node->indices) {
               compact_root->leaf_ordered_indices.push_back(indices);
            }
            
            // 3. set num of indices belonging to this leaf
            compact_node.num_faces = node->indices.size();
         } 

         // interior
         else {
            num_child_nodes = 0;
            num_child_nodes += 1u + compact_helper(node->low, index + 1);
            compact_node.high_child_offset = num_child_nodes + 1;
            compact_node.num_faces = 0;
            num_child_nodes += 1u + compact_helper(node->high, index + 1 + num_child_nodes);
         }

         return num_child_nodes;
      }
   }
   
   void bvh::compact() {
      compact_root = new compact_bvh(num_nodes);
      compact_helper(root, 0);
   }

   bvh::~bvh() {
      std::queue<bvh_node *> q;
      q.push(root);
      while (!q.empty()) {
         bvh_node *node = q.front();
         q.pop();
         if (node != nullptr) {
            q.push(node->low);
            q.push(node->high);
            delete node;
         }
      }

      delete compact_root;
   }
}
