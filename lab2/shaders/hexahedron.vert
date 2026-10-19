#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;

layout(set = 0, binding = 0) uniform Scene {
	mat4 viewProj;
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
	vec4 tint;
} modelData;

void main() {
	vec4 world = modelData.model * vec4(inPosition, 1.0);
	fragPosition = world.xyz;
	fragNormal = transpose(inverse(mat3(modelData.model))) * inNormal;
	fragColor = inColor * modelData.tint.rgb;
	gl_Position = scene.viewProj * world;
}
