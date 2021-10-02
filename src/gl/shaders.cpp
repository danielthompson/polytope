//
// Created by daniel on 12/17/20.
//

#include <fstream>
#include <glbinding/gl/gl.h>
#include "shaders.h"
#include "../common/utilities/Common.h"

// static init
std::map<std::string, std::shared_ptr<poly::shader_program>> poly::shader_cache::map = std::map<std::string, std::shared_ptr<poly::shader_program>>();

poly::shader::shader(const poly::shader_info &info) {
//   std::string shader_path = info.filepath;
//   "../resources/shaders/";
//   switch (info.type) {
//      case poly::shader_type::vert: {
//         shader_path += "vertex/";
//         break;
//      }
//      case poly::shader_type::frag: {
//         shader_path += "fragment/";
//         break;
//      }
//   }
   
   const std::string total_path = info.filepath;
   
   std::ifstream stream;
   stream.open(total_path, std::ifstream::in);
   if (stream.fail()) {
      Log.error() << "Failed to open shader file at [" << total_path << "].";
      exit(EXIT_FAILURE);
   }

   std::string contents;
   contents.assign(std::istreambuf_iterator<char>(stream),
                   std::istreambuf_iterator<char>());

   gl::GLenum gl_shader_type;
   switch (info.type) {
      case shader_type::vert: {
         gl_shader_type = gl::GL_VERTEX_SHADER;
         break;
      }
      case shader_type::frag: {
         gl_shader_type = gl::GL_FRAGMENT_SHADER;
         break;
      }
      default: {
         Log.error() << "Unknown shader type [" << info.type << "].";
         exit(EXIT_FAILURE);
      }
   }

   handle = gl::glCreateShader(gl_shader_type);
   const char *c_str = contents.c_str();
   gl::glShaderSource(handle, 1, &c_str, nullptr);
   gl::glCompileShader(handle);
   int success;
   char log_buffer[512];
   gl::glGetShaderiv(handle, gl::GL_COMPILE_STATUS, &success);
   if (!success) {
      gl::glGetShaderInfoLog(handle, 512, nullptr, log_buffer);
      Log.error() << "Failed to compile shader [" + total_path + "]:";
      Log.error() << log_buffer;
      exit(EXIT_FAILURE);
   }
}

poly::shader::~shader() {
   gl::glDeleteShader(handle);
}

poly::shader_program::shader_program(const poly::shader_program_info &info) {
   handle = gl::glCreateProgram();
   
   poly::shader* vert_shader;
   poly::shader* frag_shader;
   
   for (const auto & shader_info : info.shader_infos) {
      switch (shader_info.type) {
         case poly::shader_type::vert: {
            vert_shader = new poly::shader(shader_info);
            break;
         }
         case poly::shader_type::frag: {
            frag_shader = new poly::shader(shader_info);
            break;
         }
         default: {
            Log.error() << "Unknown shader type [" << shader_info.type << "].";
            exit(EXIT_FAILURE);
         }
      }
   }
   
   if (vert_shader != nullptr) {
      gl::glAttachShader(handle, vert_shader->handle);
   }
   if (frag_shader != nullptr) {
      gl::glAttachShader(handle, frag_shader->handle);
   }

   gl::glLinkProgram(handle);

   int success;
   char log_buffer[512];
   gl::glGetProgramiv(handle, gl::GL_LINK_STATUS, &success);
   if (!success) {
      gl::glGetProgramInfoLog(handle, 512, nullptr, log_buffer);
      Log.error() << "Failed to link program:";
      Log.error() << log_buffer;
      exit(EXIT_FAILURE);
   }

   LOG_DEBUG("Program linked.");
   
   delete vert_shader;
   delete frag_shader;
   
   if (info.uniform_names.empty()) {
      LOG_DEBUG("Program has no uniforms.");
      return;
   }
   
   for (const auto & uniform_name : info.uniform_names) {
      gl::GLint location = gl::glGetUniformLocation(handle, uniform_name.c_str());
      if (location < 0) {
         LOG_ERROR("Program doesn't have a location for uniform [" << uniform_name << "].");
         exit(EXIT_FAILURE);
      }
      
      uniform_location_map[uniform_name] = location;
   }
}

poly::shader_program::~shader_program() {
   if (gl::glIsProgram(handle)) {
      gl::glDeleteProgram(handle);
   }
}

void poly::shader_program::use() const {
   gl::glUseProgram(handle);
}

void poly::shader_program::set_uniform(const std::string &name, const gl::GLfloat value) {
   const gl::GLuint location = uniform_location_map.at(name);
   gl::glUniform1f(location, value);
}

void poly::shader_program::set_uniform(const std::string &name, gl::GLint value) {
   const gl::GLuint location = uniform_location_map.at(name);
   gl::glUniform1i(location, value);
}

void poly::shader_program::set_uniform(const std::string &name, const float v0, const float v1, const float v2) {
   const gl::GLuint location = uniform_location_map.at(name);
   gl::glUniform3f(location, v0, v1, v2);
}

void poly::shader_program::set_uniform(const std::string &name, const glm::vec4 &value) {
   const gl::GLuint location = uniform_location_map.at(name);
   gl::glUniform4fv(location, 1, &value[0]);
}

void poly::shader_program::set_uniform(const std::string &name, const glm::mat4 &value) {
   const gl::GLuint location = uniform_location_map.at(name);
   gl::glUniformMatrix4fv(location, 1, 0, &value[0][0]);
}

std::shared_ptr<poly::shader_program> poly::shader_cache::get_program(const poly::shader_program_info& program_info) {

   // TODO use hash of combined filenames as key instead of constructing a temp string

   std::string key = program_info.shader_infos[0].filepath + "|" + program_info.shader_infos[1].filepath;

   if (map.count(key) > 0) {
      LOG_DEBUG("Shader cache hit on [" << key << "].");
      return map[key];
   }

   LOG_DEBUG("Shader cache miss on [" << key << "].");

   std::shared_ptr<poly::shader_program> program = std::make_shared<poly::shader_program>(program_info);
   map[key] = program;
   return program;
}
