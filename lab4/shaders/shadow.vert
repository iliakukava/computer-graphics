#version 450

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform Scene {
	mat4 viewProj;
	mat4 lightViewProj;
	vec4 cameraPosition;
	vec4 ambient;
	vec4 dirDirection;
	vec4 dirColor;
	ivec4 counts;
} scene;

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	vec4 albedo;
	vec4 specular;
} modelData;

void main() {
	gl_Position = scene.lightViewProj * modelData.model * vec4(inPosition, 1.0);
}
