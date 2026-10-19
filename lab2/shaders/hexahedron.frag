#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Scene {
	mat4 viewProj;
	vec4 cameraPosition;
	vec4 ambient;
	vec4 dirDirection;
	vec4 dirColor;
	ivec4 counts;
} scene;

struct PointLight {
	vec4 position;
	vec4 color;
};

struct SpotLight {
	vec4 position;
	vec4 direction;
	vec4 color;
	vec4 angles;
};

layout(set = 0, binding = 1) readonly buffer PointLights {
	PointLight lights[];
} pointLights;

layout(set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight lights[];
} spotLights;

layout(set = 1, binding = 0) uniform Model {
	mat4 model;
	vec4 albedo;
	vec4 specular;
	vec4 tint;
} modelData;

vec3 blinn(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 albedo, vec3 specular, float shininess, vec3 lightColor) {
	float diffuse = max(dot(normal, lightDir), 0.0);
	vec3 halfDir = normalize(lightDir + viewDir);
	float spec = pow(max(dot(normal, halfDir), 0.0), shininess);
	return (diffuse * albedo + spec * specular) * lightColor;
}

void main() {
	vec3 normal = normalize(fragNormal);
	vec3 viewDir = normalize(scene.cameraPosition.xyz - fragPosition);
	vec3 albedo = fragColor * modelData.albedo.rgb;
	vec3 specular = modelData.specular.rgb;
	float shininess = max(modelData.specular.w, 1.0);

	vec3 color = scene.ambient.rgb * albedo;

	vec3 dirToLight = normalize(-scene.dirDirection.xyz);
	color += blinn(normal, viewDir, dirToLight, albedo, specular, shininess,
				   scene.dirColor.rgb * scene.dirColor.w);

	for (int i = 0; i < scene.counts.x; ++i) {
		vec3 toLight = pointLights.lights[i].position.xyz - fragPosition;
		float dist2 = max(dot(toLight, toLight), 0.0001);
		vec3 lightDir = toLight * inversesqrt(dist2);
		vec3 lightColor = pointLights.lights[i].color.rgb * pointLights.lights[i].color.w;
		color += blinn(normal, viewDir, lightDir, albedo, specular, shininess, lightColor) / dist2;
	}

	for (int i = 0; i < scene.counts.y; ++i) {
		vec3 toLight = spotLights.lights[i].position.xyz - fragPosition;
		float dist2 = max(dot(toLight, toLight), 0.0001);
		vec3 lightDir = toLight * inversesqrt(dist2);
		float theta = dot(-lightDir, normalize(spotLights.lights[i].direction.xyz));
		float outerCos = spotLights.lights[i].angles.y;
		float innerCos = max(spotLights.lights[i].angles.x, outerCos + 0.001);
		float spot = smoothstep(outerCos, innerCos, theta);
		vec3 lightColor = spotLights.lights[i].color.rgb * spotLights.lights[i].color.w;
		color += blinn(normal, viewDir, lightDir, albedo, specular, shininess, lightColor) * spot / dist2;
	}

	outColor = vec4(color, 1.0);
}
