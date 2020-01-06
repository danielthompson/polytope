#version 330 core

// layout(location = 0) - the buffer we use to feed the vertexPosition_modelspace attribute.
// each vertex can have numerous attributes - a position, one or more colors, one or several tex coords, etc.

// vertexPosition_modelspace - position of the vertex for each run of the vertex shader
// in - input data
layout(location = 0) in vec3 vertexPosition_modelspace;
//layout(location = 1) in vec3 vertexColor;

//out vec3 fragmentColor;

// fed in via glUniformMatrix4fv in main loop
uniform mat4 mvp;

void main() {
    // gl_Position is built-in and *must* be assigned to in the vertex shader

    gl_Position = mvp * vec4(vertexPosition_modelspace, 1);
    //fragmentColor = vertexColor;

}