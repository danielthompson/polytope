#version 330 core

//in vec3 fragmentColor;

uniform vec4 colorIn;

out vec4 color;

void main() {
//    color = fragmentColor;
    color.xyz = colorIn.xyz;
    color.a = colorIn[3];
}
