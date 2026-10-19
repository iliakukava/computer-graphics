#include "application.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>

namespace application {

namespace {

constexpr int max_lights = 16;
constexpr float floor_y = -1.0f;

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec3 color;
};

struct SceneData {
	glm::mat4 view_proj;
	glm::vec4 camera_position;
	glm::vec4 ambient;
	glm::vec4 dir_direction;
	glm::vec4 dir_color;
	glm::ivec4 counts;
};

struct ModelData {
	glm::mat4 model;
	glm::vec4 albedo;
	glm::vec4 specular;
	glm::vec4 tint;
};

struct GpuPointLight {
	glm::vec4 position;
	glm::vec4 color;
};

struct GpuSpotLight {
	glm::vec4 position;
	glm::vec4 direction;
	glm::vec4 color;
	glm::vec4 angles;
};

struct PointLight {
	glm::vec3 position;
	glm::vec3 color;
	float intensity;
};

struct SpotLight {
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 color;
	float intensity;
	float inner_degrees;
	float outer_degrees;
};

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* data = nullptr;
};

static_assert(sizeof(SceneData) == 144);
static_assert(offsetof(SceneData, counts) == 128);
static_assert(sizeof(ModelData) == 112);
static_assert(sizeof(GpuPointLight) == 32);
static_assert(sizeof(GpuSpotLight) == 64);

bool perspective = true;
float fov = 60.0f;
float ortho_size = 3.0f;
glm::vec3 position{0.0f};
glm::vec3 rotation{0.0f};
glm::vec3 scale{1.0f};
glm::vec3 tint{1.0f};

bool playing = true;
float speed = 1.0f;
float radius = 1.5f;
float height = 0.6f;
float rotation_speed = 1.0f;
float anim_time = 0.0f;

glm::vec3 camera_position{0.0f, 2.2f, 8.0f};
float camera_yaw = -90.0f;
float camera_pitch = -15.0f;

glm::vec3 ambient{0.12f, 0.12f, 0.14f};
glm::vec3 dir_direction{0.4f, -1.0f, 0.35f};
glm::vec3 dir_color{1.0f};
float dir_intensity = 1.0f;

glm::vec3 albedo{1.0f};
glm::vec3 specular{1.0f};
float shininess = 32.0f;

std::vector<PointLight> point_lights = {
	{glm::vec3(2.5f, 2.5f, 2.0f), glm::vec3(1.0f, 0.85f, 0.6f), 25.0f},
};
std::vector<SpotLight> spot_lights = {
	{glm::vec3(-2.0f, 3.5f, 2.0f), glm::vec3(0.45f, -0.8f, -0.4f), glm::vec3(0.6f, 0.75f, 1.0f), 40.0f, 15.0f, 28.0f},
};

Buffer vertex_buffer;
Buffer index_buffer;
Buffer room_vertex_buffer;
Buffer room_index_buffer;
Buffer scene_buffer;
Buffer model_buffer;
Buffer room_buffer;
Buffer point_buffer;
Buffer spot_buffer;
uint32_t index_count = 0;
uint32_t room_index_count = 0;

VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet model_set = VK_NULL_HANDLE;
VkDescriptorSet room_set = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

std::vector<char> readFile(const char* path) {
	std::ifstream file(path, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open " << path << '\n';
		return {};
	}

	size_t size = size_t(file.tellg());
	std::vector<char> bytes(size);
	file.seekg(0);
	file.read(bytes.data(), size);
	return bytes;
}

VkShaderModule createShaderModule(const char* path) {
	std::vector<char> code = readFile(path);
	if (code.empty()) {
		return VK_NULL_HANDLE;
	}

	VkShaderModuleCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	info.codeSize = code.size();
	info.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module\n";
		return VK_NULL_HANDLE;
	}
	return module;
}

Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = size;
	buffer_info.usage = usage;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
	alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
					   VMA_ALLOCATION_CREATE_MAPPED_BIT;

	Buffer result;
	VmaAllocationInfo info{};
	if (vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info,
						&result.buffer, &result.allocation, &info) != VK_SUCCESS) {
		std::cerr << "Failed to create buffer\n";
		return {};
	}

	result.data = info.pMappedData;
	return result;
}

void destroyBuffer(Buffer& buffer) {
	if (buffer.buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(graphics::internal::context.allocator, buffer.buffer, buffer.allocation);
	}
	buffer = {};
}

bool createCube() {
	const glm::vec3 corners[8] = {
		{-1.0f, -1.0f, -1.0f},
		{ 1.0f, -1.0f, -1.0f},
		{ 1.0f,  1.0f, -1.0f},
		{-1.0f,  1.0f, -1.0f},
		{-1.0f, -1.0f,  1.0f},
		{ 1.0f, -1.0f,  1.0f},
		{ 1.0f,  1.0f,  1.0f},
		{-1.0f,  1.0f,  1.0f},
	};

	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;

	auto addFace = [&](int a, int b, int c, int d) {
		glm::vec3 normal = glm::normalize(glm::cross(corners[b] - corners[a], corners[c] - corners[a]));
		uint16_t base = uint16_t(vertices.size());
		int ids[4] = {a, b, c, d};
		for (int id : ids) {
			Vertex vertex;
			vertex.position = corners[id];
			vertex.normal = normal;
			vertex.color = glm::vec3(1.0f);
			vertices.push_back(vertex);
		}
		indices.insert(indices.end(), {base, uint16_t(base + 1), uint16_t(base + 2),
										base, uint16_t(base + 2), uint16_t(base + 3)});
	};

	addFace(4, 5, 6, 7);
	addFace(1, 0, 3, 2);
	addFace(0, 4, 7, 3);
	addFace(5, 1, 2, 6);
	addFace(0, 1, 5, 4);
	addFace(3, 7, 6, 2);

	index_count = uint32_t(indices.size());
	vertex_buffer = createBuffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	index_buffer = createBuffer(sizeof(uint16_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	if (vertex_buffer.data == nullptr || index_buffer.data == nullptr) {
		return false;
	}

	std::memcpy(vertex_buffer.data, vertices.data(), sizeof(Vertex) * vertices.size());
	std::memcpy(index_buffer.data, indices.data(), sizeof(uint16_t) * indices.size());
	vmaFlushAllocation(graphics::internal::context.allocator, vertex_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(graphics::internal::context.allocator, index_buffer.allocation, 0, VK_WHOLE_SIZE);
	return true;
}

bool uploadMesh(Buffer& vertices_out, Buffer& indices_out, uint32_t& index_count_out,
				const std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices) {
	vertices_out = createBuffer(sizeof(Vertex) * vertices.size(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	indices_out = createBuffer(sizeof(uint16_t) * indices.size(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	if (vertices_out.data == nullptr || indices_out.data == nullptr) {
		return false;
	}

	std::memcpy(vertices_out.data, vertices.data(), sizeof(Vertex) * vertices.size());
	std::memcpy(indices_out.data, indices.data(), sizeof(uint16_t) * indices.size());
	vmaFlushAllocation(graphics::internal::context.allocator, vertices_out.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(graphics::internal::context.allocator, indices_out.allocation, 0, VK_WHOLE_SIZE);
	index_count_out = uint32_t(indices.size());
	return true;
}

bool createRoom() {
	const float extent = 5.0f;
	const float wall_top = 4.0f;

	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
	auto addQuad = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
		glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
		uint16_t base = uint16_t(vertices.size());
		glm::vec3 positions[4] = {a, b, c, d};
		for (glm::vec3 point : positions) {
			Vertex vertex;
			vertex.position = point;
			vertex.normal = normal;
			vertex.color = glm::vec3(1.0f);
			vertices.push_back(vertex);
		}
		indices.insert(indices.end(), {base, uint16_t(base + 1), uint16_t(base + 2),
										base, uint16_t(base + 2), uint16_t(base + 3)});
	};

	addQuad({-extent, floor_y, -extent}, {-extent, floor_y, extent},
			{extent, floor_y, extent}, {extent, floor_y, -extent});
	addQuad({-extent, floor_y, -extent}, {extent, floor_y, -extent},
			{extent, wall_top, -extent}, {-extent, wall_top, -extent});
	addQuad({-extent, floor_y, -extent}, {-extent, wall_top, -extent},
			{-extent, wall_top, extent}, {-extent, floor_y, extent});
	addQuad({extent, floor_y, extent}, {extent, wall_top, extent},
			{extent, wall_top, -extent}, {extent, floor_y, -extent});

	return uploadMesh(room_vertex_buffer, room_index_buffer, room_index_count, vertices, indices);
}

bool createDescriptors() {
	auto& device = graphics::internal::context.device;

	VkDescriptorSetLayoutBinding scene_bindings[3]{};
	scene_bindings[0].binding = 0;
	scene_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	scene_bindings[0].descriptorCount = 1;
	scene_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	scene_bindings[1].binding = 1;
	scene_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	scene_bindings[1].descriptorCount = 1;
	scene_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	scene_bindings[2].binding = 2;
	scene_bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	scene_bindings[2].descriptorCount = 1;
	scene_bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo scene_info{};
	scene_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	scene_info.bindingCount = 3;
	scene_info.pBindings = scene_bindings;
	if (vkCreateDescriptorSetLayout(device, &scene_info, nullptr, &scene_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create scene descriptor set layout\n";
		return false;
	}

	VkDescriptorSetLayoutBinding model_binding{};
	model_binding.binding = 0;
	model_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	model_binding.descriptorCount = 1;
	model_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo model_info{};
	model_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	model_info.bindingCount = 1;
	model_info.pBindings = &model_binding;
	if (vkCreateDescriptorSetLayout(device, &model_info, nullptr, &model_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create model descriptor set layout\n";
		return false;
	}

	VkDescriptorPoolSize pool_sizes[2]{};
	pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pool_sizes[0].descriptorCount = 3;
	pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pool_sizes[1].descriptorCount = 2;

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets = 3;
	pool_info.poolSizeCount = 2;
	pool_info.pPoolSizes = pool_sizes;
	if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetLayout layouts[3] = {scene_layout, model_layout, model_layout};
	VkDescriptorSetAllocateInfo alloc_info{};
	alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc_info.descriptorPool = descriptor_pool;
	alloc_info.descriptorSetCount = 3;
	alloc_info.pSetLayouts = layouts;

	VkDescriptorSet sets[3] = {};
	if (vkAllocateDescriptorSets(device, &alloc_info, sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}
	scene_set = sets[0];
	model_set = sets[1];
	room_set = sets[2];

	VkDescriptorBufferInfo scene_info_buffer{};
	scene_info_buffer.buffer = scene_buffer.buffer;
	scene_info_buffer.range = sizeof(SceneData);
	VkDescriptorBufferInfo point_info{};
	point_info.buffer = point_buffer.buffer;
	point_info.range = sizeof(GpuPointLight) * max_lights;
	VkDescriptorBufferInfo spot_info{};
	spot_info.buffer = spot_buffer.buffer;
	spot_info.range = sizeof(GpuSpotLight) * max_lights;
	VkDescriptorBufferInfo model_info_buffer{};
	model_info_buffer.buffer = model_buffer.buffer;
	model_info_buffer.range = sizeof(ModelData);
	VkDescriptorBufferInfo room_info_buffer{};
	room_info_buffer.buffer = room_buffer.buffer;
	room_info_buffer.range = sizeof(ModelData);

	VkWriteDescriptorSet writes[5]{};
	writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[0].dstSet = scene_set;
	writes[0].dstBinding = 0;
	writes[0].descriptorCount = 1;
	writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[0].pBufferInfo = &scene_info_buffer;
	writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[1].dstSet = scene_set;
	writes[1].dstBinding = 1;
	writes[1].descriptorCount = 1;
	writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[1].pBufferInfo = &point_info;
	writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[2].dstSet = scene_set;
	writes[2].dstBinding = 2;
	writes[2].descriptorCount = 1;
	writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writes[2].pBufferInfo = &spot_info;
	writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[3].dstSet = model_set;
	writes[3].dstBinding = 0;
	writes[3].descriptorCount = 1;
	writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[3].pBufferInfo = &model_info_buffer;
	writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writes[4].dstSet = room_set;
	writes[4].dstBinding = 0;
	writes[4].descriptorCount = 1;
	writes[4].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writes[4].pBufferInfo = &room_info_buffer;
	vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
	return true;
}

bool createPipeline() {
	auto& device = graphics::internal::context.device;

	VkShaderModule vert = createShaderModule("shaders/hexahedron.vert.spv");
	VkShaderModule frag = createShaderModule("shaders/hexahedron.frag.spv");
	if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE) {
		vkDestroyShaderModule(device, vert, nullptr);
		vkDestroyShaderModule(device, frag, nullptr);
		return false;
	}

	VkPipelineShaderStageCreateInfo stages[2]{};
	stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
	stages[0].module = vert;
	stages[0].pName = "main";
	stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	stages[1].module = frag;
	stages[1].pName = "main";

	VkVertexInputBindingDescription binding{};
	binding.binding = 0;
	binding.stride = sizeof(Vertex);
	binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	VkVertexInputAttributeDescription attributes[3]{};
	attributes[0].location = 0;
	attributes[0].binding = 0;
	attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[0].offset = offsetof(Vertex, position);
	attributes[1].location = 1;
	attributes[1].binding = 0;
	attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[1].offset = offsetof(Vertex, normal);
	attributes[2].location = 2;
	attributes[2].binding = 0;
	attributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[2].offset = offsetof(Vertex, color);

	VkPipelineVertexInputStateCreateInfo vertex_input{};
	vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input.vertexBindingDescriptionCount = 1;
	vertex_input.pVertexBindingDescriptions = &binding;
	vertex_input.vertexAttributeDescriptionCount = 3;
	vertex_input.pVertexAttributeDescriptions = attributes;

	VkPipelineInputAssemblyStateCreateInfo input_assembly{};
	input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
	input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

	VkPipelineViewportStateCreateInfo viewport{};
	viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
	viewport.viewportCount = 1;
	viewport.scissorCount = 1;

	VkPipelineRasterizationStateCreateInfo rasterization{};
	rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
	rasterization.polygonMode = VK_POLYGON_MODE_FILL;
	rasterization.cullMode = VK_CULL_MODE_BACK_BIT;
	rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterization.lineWidth = 1.0f;

	VkPipelineMultisampleStateCreateInfo multisample{};
	multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

	VkPipelineDepthStencilStateCreateInfo depth{};
	depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	depth.depthTestEnable = VK_TRUE;
	depth.depthWriteEnable = VK_TRUE;
	depth.depthCompareOp = VK_COMPARE_OP_LESS;

	VkPipelineColorBlendAttachmentState color_attachment{};
	color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
									  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

	VkPipelineColorBlendStateCreateInfo color_blend{};
	color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
	color_blend.attachmentCount = 1;
	color_blend.pAttachments = &color_attachment;

	VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic{};
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dynamic_states;

	VkDescriptorSetLayout set_layouts[2] = {scene_layout, model_layout};
	VkPipelineLayoutCreateInfo layout_info{};
	layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout_info.setLayoutCount = 2;
	layout_info.pSetLayouts = set_layouts;
	if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		vkDestroyShaderModule(device, vert, nullptr);
		vkDestroyShaderModule(device, frag, nullptr);
		return false;
	}

	VkGraphicsPipelineCreateInfo pipeline_info{};
	pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
	pipeline_info.stageCount = 2;
	pipeline_info.pStages = stages;
	pipeline_info.pVertexInputState = &vertex_input;
	pipeline_info.pInputAssemblyState = &input_assembly;
	pipeline_info.pViewportState = &viewport;
	pipeline_info.pRasterizationState = &rasterization;
	pipeline_info.pMultisampleState = &multisample;
	pipeline_info.pDepthStencilState = &depth;
	pipeline_info.pColorBlendState = &color_blend;
	pipeline_info.pDynamicState = &dynamic;
	pipeline_info.layout = pipeline_layout;
	pipeline_info.renderPass = graphics::internal::context.render_pass;

	VkResult result = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
	vkDestroyShaderModule(device, vert, nullptr);
	vkDestroyShaderModule(device, frag, nullptr);
	if (result != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}
	return true;
}

glm::mat4 modelMatrix() {
	glm::vec3 offset;
	offset.x = radius * std::sin(anim_time);
	offset.y = height * std::sin(anim_time);
	offset.z = radius * std::sin(2.0f * anim_time);

	glm::vec3 angles = rotation;
	angles.x += glm::degrees(0.5f * anim_time * rotation_speed);
	angles.y += glm::degrees(anim_time * rotation_speed);

	glm::mat4 model(1.0f);
	model = glm::translate(model, position + offset);
	model = glm::rotate(model, glm::radians(angles.y), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::rotate(model, glm::radians(angles.x), glm::vec3(1.0f, 0.0f, 0.0f));
	model = glm::rotate(model, glm::radians(angles.z), glm::vec3(0.0f, 0.0f, 1.0f));
	model = glm::scale(model, scale);

	float lowest = model[3].y;
	for (float x : {-1.0f, 1.0f}) {
		for (float y : {-1.0f, 1.0f}) {
			for (float z : {-1.0f, 1.0f}) {
				lowest = std::min(lowest, (model * glm::vec4(x, y, z, 1.0f)).y);
			}
		}
	}
	if (lowest < floor_y) {
		model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, floor_y - lowest, 0.0f)) * model;
	}
	return model;
}

glm::mat4 viewMatrix() {
	float yaw = glm::radians(camera_yaw);
	float pitch = glm::radians(camera_pitch);
	glm::vec3 front;
	front.x = std::cos(yaw) * std::cos(pitch);
	front.y = std::sin(pitch);
	front.z = std::sin(yaw) * std::cos(pitch);
	return glm::lookAt(camera_position, camera_position + glm::normalize(front), glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 projectionMatrix(float aspect) {
	glm::mat4 projection;
	if (perspective) {
		projection = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.0f);
	} else {
		float half_width = ortho_size * aspect;
		projection = glm::ortho(-half_width, half_width, -ortho_size, ortho_size, 0.1f, 100.0f);
	}

	projection[1][1] *= -1.0f;
	return projection;
}

void drawLightList() {
	if (ImGui::CollapsingHeader("Точечные источники", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (point_lights.size() < max_lights && ImGui::Button("Добавить точечный")) {
			point_lights.push_back({glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(1.0f), 20.0f});
		}

		int remove = -1;
		for (int i = 0; i < int(point_lights.size()); ++i) {
			ImGui::PushID(i);
			ImGui::Text("Источник %d", i + 1);
			ImGui::DragFloat3("Позиция", &point_lights[i].position.x, 0.05f);
			ImGui::ColorEdit3("Цвет", &point_lights[i].color.x);
			ImGui::SliderFloat("Интенсивность", &point_lights[i].intensity, 0.0f, 100.0f);
			if (ImGui::Button("Удалить")) {
				remove = i;
			}
			ImGui::Separator();
			ImGui::PopID();
		}
		if (remove >= 0) {
			point_lights.erase(point_lights.begin() + remove);
		}
	}

	if (ImGui::CollapsingHeader("Прожекторы", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (spot_lights.size() < max_lights && ImGui::Button("Добавить прожектор")) {
			spot_lights.push_back({glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f), 30.0f, 12.0f, 25.0f});
		}

		int remove = -1;
		for (int i = 0; i < int(spot_lights.size()); ++i) {
			ImGui::PushID(100 + i);
			ImGui::Text("Прожектор %d", i + 1);
			ImGui::DragFloat3("Позиция", &spot_lights[i].position.x, 0.05f);
			ImGui::DragFloat3("Направление", &spot_lights[i].direction.x, 0.02f);
			ImGui::ColorEdit3("Цвет", &spot_lights[i].color.x);
			ImGui::SliderFloat("Интенсивность", &spot_lights[i].intensity, 0.0f, 100.0f);
			ImGui::SliderFloat("Внутренний угол", &spot_lights[i].inner_degrees, 1.0f, 60.0f);
			ImGui::SliderFloat("Внешний угол", &spot_lights[i].outer_degrees, 1.0f, 80.0f);
			if (ImGui::Button("Удалить")) {
				remove = i;
			}
			ImGui::Separator();
			ImGui::PopID();
		}
		if (remove >= 0) {
			spot_lights.erase(spot_lights.begin() + remove);
		}
	}
}

} // namespace

bool initialize() {
	ImFont* font = ImGui::GetIO().Fonts->AddFontFromFileTTF(
		"C:/Windows/Fonts/segoeui.ttf", 18.0f, nullptr, ImGui::GetIO().Fonts->GetGlyphRangesCyrillic());
	if (font != nullptr) {
		ImGui::GetIO().FontDefault = font;
	}

	scene_buffer = createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	model_buffer = createBuffer(sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	room_buffer = createBuffer(sizeof(ModelData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	point_buffer = createBuffer(sizeof(GpuPointLight) * max_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	spot_buffer = createBuffer(sizeof(GpuSpotLight) * max_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	if (!createCube() || !createRoom() || scene_buffer.data == nullptr || model_buffer.data == nullptr ||
		room_buffer.data == nullptr || point_buffer.data == nullptr || spot_buffer.data == nullptr ||
		!createDescriptors() || !createPipeline()) {
		shutdown();
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_layout, nullptr);
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	model_layout = VK_NULL_HANDLE;
	scene_layout = VK_NULL_HANDLE;

	destroyBuffer(spot_buffer);
	destroyBuffer(point_buffer);
	destroyBuffer(room_buffer);
	destroyBuffer(model_buffer);
	destroyBuffer(scene_buffer);
	destroyBuffer(room_index_buffer);
	destroyBuffer(room_vertex_buffer);
	destroyBuffer(index_buffer);
	destroyBuffer(vertex_buffer);
}

void update(double time) {
	static double last_time = time;
	float dt = float(time - last_time);
	last_time = time;
	if (playing) {
		anim_time += dt * speed;
	}

	ImGui::SetNextWindowSize(ImVec2(420.0f, 640.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Лабораторная 2");
	ImGui::TextUnformatted("Вариант 12");

	if (ImGui::CollapsingHeader("Проекция", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (ImGui::RadioButton("Перспективная", perspective)) {
			perspective = true;
		}
		ImGui::SameLine();
		if (ImGui::RadioButton("Ортографическая", !perspective)) {
			perspective = false;
		}
		if (perspective) {
			ImGui::SliderFloat("FOV", &fov, 20.0f, 100.0f);
		} else {
			ImGui::SliderFloat("Размер", &ortho_size, 1.0f, 8.0f);
		}
	}

	if (ImGui::CollapsingHeader("Фигура", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Позиция", &position.x, 0.02f);
		ImGui::DragFloat3("Поворот", &rotation.x, 0.5f);
		ImGui::DragFloat3("Масштаб", &scale.x, 0.01f, 0.05f, 5.0f);
		ImGui::ColorEdit3("Цвет", &tint.x);
	}

	if (ImGui::CollapsingHeader("Анимация")) {
		if (ImGui::Button(playing ? "Пауза" : "Воспроизвести")) {
			playing = !playing;
		}
		ImGui::SliderFloat("Скорость", &speed, 0.0f, 4.0f);
		ImGui::SliderFloat("Радиус", &radius, 0.0f, 3.0f);
		ImGui::SliderFloat("Высота", &height, 0.0f, 2.0f);
		ImGui::SliderFloat("Скорость вращения", &rotation_speed, 0.0f, 4.0f);
	}

	if (ImGui::CollapsingHeader("Камера", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Позиция камеры", &camera_position.x, 0.05f);
		ImGui::SliderFloat("Поворот", &camera_yaw, -180.0f, 180.0f);
		ImGui::SliderFloat("Наклон", &camera_pitch, -89.0f, 89.0f);
	}

	if (ImGui::CollapsingHeader("Направленный свет", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Фоновый свет", &ambient.x);
		ImGui::DragFloat3("Направление лучей", &dir_direction.x, 0.02f);
		ImGui::ColorEdit3("Цвет света", &dir_color.x);
		ImGui::SliderFloat("Интенсивность", &dir_intensity, 0.0f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Материал", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Альбедо", &albedo.x);
		ImGui::ColorEdit3("Цвет блика", &specular.x);
		ImGui::SliderFloat("Коэффициент блика", &shininess, 1.0f, 128.0f);
	}

	drawLightList();
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	if (fd.command_buffer == VK_NULL_HANDLE) {
		return;
	}

	auto& context = graphics::internal::context;
	VkCommandBuffer cmd = fd.command_buffer;
	VkExtent2D extent = context.swapchain_extent;
	float aspect = extent.height == 0 ? 1.0f : float(extent.width) / float(extent.height);

	SceneData scene;
	scene.view_proj = projectionMatrix(aspect) * viewMatrix();
	scene.camera_position = glm::vec4(camera_position, 1.0f);
	scene.ambient = glm::vec4(ambient, 1.0f);
	scene.dir_direction = glm::vec4(dir_direction, 0.0f);
	scene.dir_color = glm::vec4(dir_color, dir_intensity);
	scene.counts = glm::ivec4(int(point_lights.size()), int(spot_lights.size()), 0, 0);
	std::memcpy(scene_buffer.data, &scene, sizeof(scene));

	ModelData model;
	model.model = modelMatrix();
	model.albedo = glm::vec4(albedo, 1.0f);
	model.specular = glm::vec4(specular, shininess);
	model.tint = glm::vec4(tint, 1.0f);
	std::memcpy(model_buffer.data, &model, sizeof(model));

	ModelData room;
	room.model = glm::mat4(1.0f);
	room.albedo = glm::vec4(0.22f, 0.22f, 0.24f, 1.0f);
	room.specular = glm::vec4(0.04f, 0.04f, 0.04f, 16.0f);
	room.tint = glm::vec4(1.0f);
	std::memcpy(room_buffer.data, &room, sizeof(room));

	GpuPointLight gpu_points[max_lights]{};
	for (size_t i = 0; i < point_lights.size(); ++i) {
		gpu_points[i].position = glm::vec4(point_lights[i].position, 1.0f);
		gpu_points[i].color = glm::vec4(point_lights[i].color, point_lights[i].intensity);
	}
	std::memcpy(point_buffer.data, gpu_points, sizeof(gpu_points));

	GpuSpotLight gpu_spots[max_lights]{};
	for (size_t i = 0; i < spot_lights.size(); ++i) {
		float inner = spot_lights[i].inner_degrees;
		float outer = spot_lights[i].outer_degrees;
		if (outer < inner) {
			outer = inner;
		}
		gpu_spots[i].position = glm::vec4(spot_lights[i].position, 1.0f);
		gpu_spots[i].direction = glm::vec4(spot_lights[i].direction, 0.0f);
		gpu_spots[i].color = glm::vec4(spot_lights[i].color, spot_lights[i].intensity);
		gpu_spots[i].angles = glm::vec4(std::cos(glm::radians(inner)), std::cos(glm::radians(outer)), 0.0f, 0.0f);
	}
	std::memcpy(spot_buffer.data, gpu_spots, sizeof(gpu_spots));

	vmaFlushAllocation(context.allocator, scene_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(context.allocator, model_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(context.allocator, room_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(context.allocator, point_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(context.allocator, spot_buffer.allocation, 0, VK_WHOLE_SIZE);

	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo begin{};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &begin);

	VkClearValue clears[2]{};
	clears[0].color.float32[0] = 0.08f;
	clears[0].color.float32[1] = 0.09f;
	clears[0].color.float32[2] = 0.11f;
	clears[0].color.float32[3] = 1.0f;
	clears[1].depthStencil.depth = 1.0f;

	VkRenderPassBeginInfo render_pass{};
	render_pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	render_pass.renderPass = context.render_pass;
	render_pass.framebuffer = fd.framebuffer;
	render_pass.renderArea.extent = extent;
	render_pass.clearValueCount = 2;
	render_pass.pClearValues = clears;
	vkCmdBeginRenderPass(cmd, &render_pass, VK_SUBPASS_CONTENTS_INLINE);

	VkViewport viewport{};
	viewport.width = float(extent.width);
	viewport.height = float(extent.height);
	viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	VkRect2D scissor{};
	scissor.extent = extent;
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	VkDescriptorSet room_sets[2] = {scene_set, room_set};
	VkDescriptorSet cube_sets[2] = {scene_set, model_set};
	VkDeviceSize offset = 0;
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	vkCmdBindVertexBuffers(cmd, 0, 1, &room_vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, room_index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 2, room_sets, 0, nullptr);
	vkCmdDrawIndexed(cmd, room_index_count, 1, 0, 0, 0);

	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 2, cube_sets, 0, nullptr);
	vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
