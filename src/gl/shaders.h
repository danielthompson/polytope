//
// Created by daniel on 12/17/20.
//

#ifndef UBERTUXRACER_SHADERS_H
#define UBERTUXRACER_SHADERS_H

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <glbinding/gl/gl.h>
#include <glm/glm.hpp>

namespace poly {
   enum shader_type {
      vert,
      frag
   };

   class shader_info {
   public:
      std::string filepath;
      poly::shader_type type;
   };

   class shader_program_info {
   public:
      std::vector<poly::shader_info> shader_infos;
      std::vector<std::string> uniform_names;
      shader_program_info(const std::vector<poly::shader_info> &shader_infos,
                          const std::vector<std::string> uniform_names)
            : shader_infos(shader_infos), uniform_names(uniform_names) { }
   };
   
   class shader_program;
   
   class shader {
   private:
      unsigned int handle;
      explicit shader(const poly::shader_info& info);
      ~shader();
      
      friend class shader_program;
   };
   
   class shader_program {
   public:
      explicit shader_program(const poly::shader_program_info &info);
      ~shader_program();
      unsigned int handle;
      void use() const;
      
      void set_uniform(const std::string &name, gl::GLfloat value);
      void set_uniform(const std::string &name, gl::GLint value);
      void set_uniform(const std::string &name, float v0, float v1, float v2);
      void set_uniform(const std::string &name, const glm::vec4& value);
      void set_uniform(const std::string &name, const glm::mat4& value);
      
      std::map<std::string, gl::GLuint> uniform_location_map;
   };
   
   class shader_cache {
   public:
      static std::shared_ptr<poly::shader_program> get_program(const poly::shader_program_info& shader_info);
      
   private:
      static std::map<std::string, std::shared_ptr<poly::shader_program>> map;
   };
}

#endif //UBERTUXRACER_SHADERS_H
