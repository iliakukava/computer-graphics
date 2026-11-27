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

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace application {

namespace {

constexpr int max_lights = 16;
constexpr int texture_count = 3;
constexpr int model_count = 2;
constexpr uint32_t shadow_map_size = 2048;
constexpr float floor_y = -1.0f;

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

struct SceneData {
	glm::mat4 view_proj;
	glm::mat4 light_view_proj;
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
};

struct PushConstants {
	float time;
	int use_effect;
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

struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	VkImageView view = VK_NULL_HANDLE;
};

static_assert(sizeof(SceneData) == 208);
static_assert(sizeof(ModelData) == 96);
static_assert(sizeof(PushConstants) == 8);

const char* texture_paths[texture_count] = {
	"textures/checker.png",
	"textures/brick.png",
	"textures/grid.png",
};

const char* texture_names[texture_count] = {
	"Клетка",
	"Кирпич",
	"Сетка",
};

bool perspective = true;
float fov = 60.0f;
float ortho_size = 3.0f;
glm::vec3 position{0.0f, 0.2f, 0.0f};
glm::vec3 rotation{0.0f};
glm::vec3 scale{1.0f};

bool playing = true;
float speed = 1.0f;
float radius = 1.4f;
float height = 0.6f;
float rotation_speed = 1.0f;
float anim_time = 0.0f;
float current_time = 0.0f;

int cube_texture = 0;
int room_texture = 1;
bool use_effect = false;

float shadow_area = 7.5f;
float depth_bias_constant = 1.25f;
float depth_bias_slope = 1.75f;

glm::vec3 camera_position{0.0f, 3.0f, 8.2f};
float camera_yaw = -90.0f;
float camera_pitch = -18.0f;

glm::vec3 ambient{0.20f, 0.20f, 0.22f};
glm::vec3 dir_direction{-0.55f, -1.0f, -0.45f};
glm::vec3 dir_color{1.0f, 0.97f, 0.9f};
float dir_intensity = 1.3f;

glm::vec3 albedo{1.0f};
glm::vec3 specular{1.0f};
float shininess = 32.0f;

std::vector<PointLight> point_lights = {
	{glm::vec3(2.5f, 2.5f, 2.0f), glm::vec3(1.0f, 0.85f, 0.6f), 18.0f},
};
std::vector<SpotLight> spot_lights = {
	{glm::vec3(-2.0f, 3.5f, 2.0f), glm::vec3(0.45f, -0.8f, -0.4f), glm::vec3(0.6f, 0.75f, 1.0f), 30.0f, 15.0f, 28.0f},
};

Buffer vertex_buffer;
Buffer index_buffer;
Buffer room_vertex_buffer;
Buffer room_index_buffer;
Buffer scene_buffer;
Buffer model_buffer;
Buffer point_buffer;
Buffer spot_buffer;
uint32_t index_count = 0;
uint32_t room_index_count = 0;
VkDeviceSize model_stride = 256;

Texture textures[texture_count];
VkSampler texture_sampler = VK_NULL_HANDLE;
VkCommandPool transfer_pool = VK_NULL_HANDLE;

VkImage shadow_image = VK_NULL_HANDLE;
VmaAllocation shadow_allocation = VK_NULL_HANDLE;
VkImageView shadow_view = VK_NULL_HANDLE;
VkSampler shadow_sampler = VK_NULL_HANDLE;

VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet scene_set = VK_NULL_HANDLE;
VkDescriptorSet material_sets[texture_count] = {};
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;
VkPipeline shadow_pipeline = VK_NULL_HANDLE;

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

Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, bool gpu_only = false) {
	VkBufferCreateInfo buffer_info{};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size = size;
	buffer_info.usage = usage;

	VmaAllocationCreateInfo alloc_info{};
	if (gpu_only) {
		alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	} else {
		alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
		alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
						   VMA_ALLOCATION_CREATE_MAPPED_BIT;
	}

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

VkCommandBuffer beginCommands() {
	VkCommandBufferAllocateInfo alloc_info{};
	alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	alloc_info.commandPool = transfer_pool;
	alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	alloc_info.commandBufferCount = 1;

	VkCommandBuffer cmd = VK_NULL_HANDLE;
	vkAllocateCommandBuffers(graphics::internal::context.device, &alloc_info, &cmd);

	VkCommandBufferBeginInfo begin_info{};
	begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &begin_info);
	return cmd;
}

void endCommands(VkCommandBuffer cmd) {
	auto& context = graphics::internal::context;
	vkEndCommandBuffer(cmd);

	VkSubmitInfo submit{};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;
	vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
	vkQueueWaitIdle(context.graphics_queue);

	vkFreeCommandBuffers(context.device, transfer_pool, 1, &cmd);
}

Buffer createGpuBuffer(const void* src, VkDeviceSize size, VkBufferUsageFlags usage) {
	Buffer staging = createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	if (staging.data == nullptr) {
		return {};
	}
	std::memcpy(staging.data, src, size);
	vmaFlushAllocation(graphics::internal::context.allocator, staging.allocation, 0, VK_WHOLE_SIZE);

	Buffer gpu = createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
	if (gpu.buffer == VK_NULL_HANDLE) {
		destroyBuffer(staging);
		return {};
	}

	VkCommandBuffer cmd = beginCommands();
	VkBufferCopy copy{};
	copy.size = size;
	vkCmdCopyBuffer(cmd, staging.buffer, gpu.buffer, 1, &copy);
	endCommands(cmd);

	destroyBuffer(staging);
	return gpu;
}

bool uploadMesh(Buffer& vertices_out, Buffer& indices_out, uint32_t& index_count_out,
				const std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices) {
	vertices_out = createGpuBuffer(vertices.data(), sizeof(Vertex) * vertices.size(),
								   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	indices_out = createGpuBuffer(indices.data(), sizeof(uint16_t) * indices.size(),
								  VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	index_count_out = uint32_t(indices.size());
	return vertices_out.buffer != VK_NULL_HANDLE && indices_out.buffer != VK_NULL_HANDLE;
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
	const glm::vec2 uvs[4] = {
		{0.0f, 1.0f},
		{1.0f, 1.0f},
		{1.0f, 0.0f},
		{0.0f, 0.0f},
	};

	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;

	auto addFace = [&](int a, int b, int c, int d) {
		glm::vec3 normal = glm::normalize(glm::cross(corners[b] - corners[a], corners[c] - corners[a]));
		uint16_t base = uint16_t(vertices.size());
		int ids[4] = {a, b, c, d};
		for (int i = 0; i < 4; ++i) {
			Vertex vertex;
			vertex.position = corners[ids[i]];
			vertex.normal = normal;
			vertex.uv = uvs[i];
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

	return uploadMesh(vertex_buffer, index_buffer, index_count, vertices, indices);
}

bool createRoom() {
	const float extent = 5.0f;
	const float wall_top = 4.0f;
	const glm::vec2 uvs[4] = {
		{0.0f, 4.0f},
		{4.0f, 4.0f},
		{4.0f, 0.0f},
		{0.0f, 0.0f},
	};

	std::vector<Vertex> vertices;
	std::vector<uint16_t> indices;
	auto addQuad = [&](glm::vec3 a, glm::vec3 b, glm::vec3 c, glm::vec3 d) {
		glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
		uint16_t base = uint16_t(vertices.size());
		glm::vec3 positions[4] = {a, b, c, d};
		for (int i = 0; i < 4; ++i) {
			Vertex vertex;
			vertex.position = positions[i];
			vertex.normal = normal;
			vertex.uv = uvs[i];
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

bool loadTexture(const char* path, Texture& out) {
	int width = 0;
	int height = 0;
	int channels = 0;
	stbi_uc* pixels = stbi_load(path, &width, &height, &channels, STBI_rgb_alpha);
	if (pixels == nullptr) {
		std::cerr << "Failed to load texture " << path << '\n';
		return false;
	}

	VkDeviceSize image_size = VkDeviceSize(width) * VkDeviceSize(height) * 4;
	Buffer staging = createBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
	if (staging.data == nullptr) {
		stbi_image_free(pixels);
		return false;
	}
	std::memcpy(staging.data, pixels, image_size);
	vmaFlushAllocation(graphics::internal::context.allocator, staging.allocation, 0, VK_WHOLE_SIZE);
	stbi_image_free(pixels);

	VkImageCreateInfo image_info{};
	image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType = VK_IMAGE_TYPE_2D;
	image_info.extent.width = uint32_t(width);
	image_info.extent.height = uint32_t(height);
	image_info.extent.depth = 1;
	image_info.mipLevels = 1;
	image_info.arrayLayers = 1;
	image_info.format = VK_FORMAT_R8G8B8A8_SRGB;
	image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	image_info.samples = VK_SAMPLE_COUNT_1_BIT;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateImage(graphics::internal::context.allocator, &image_info, &alloc_info,
					   &out.image, &out.allocation, nullptr) != VK_SUCCESS) {
		destroyBuffer(staging);
		return false;
	}

	VkCommandBuffer cmd = beginCommands();

	VkImageMemoryBarrier to_dst{};
	to_dst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	to_dst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	to_dst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	to_dst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_dst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_dst.image = out.image;
	to_dst.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	to_dst.subresourceRange.levelCount = 1;
	to_dst.subresourceRange.layerCount = 1;
	to_dst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
						 0, 0, nullptr, 0, nullptr, 1, &to_dst);

	VkBufferImageCopy region{};
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.layerCount = 1;
	region.imageExtent = {uint32_t(width), uint32_t(height), 1};
	vkCmdCopyBufferToImage(cmd, staging.buffer, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	VkImageMemoryBarrier to_shader = to_dst;
	to_shader.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	to_shader.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
						 0, 0, nullptr, 0, nullptr, 1, &to_shader);

	endCommands(cmd);
	destroyBuffer(staging);

	VkImageViewCreateInfo view_info{};
	view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_info.image = out.image;
	view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
	view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	view_info.subresourceRange.levelCount = 1;
	view_info.subresourceRange.layerCount = 1;
	return vkCreateImageView(graphics::internal::context.device, &view_info, nullptr, &out.view) == VK_SUCCESS;
}

bool createTextures() {
	VkSamplerCreateInfo sampler_info{};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	if (vkCreateSampler(graphics::internal::context.device, &sampler_info, nullptr, &texture_sampler) != VK_SUCCESS) {
		return false;
	}

	for (int i = 0; i < texture_count; ++i) {
		if (!loadTexture(texture_paths[i], textures[i])) {
			return false;
		}
	}
	return true;
}

// текстура глубины для карты теней и сэмплер со сравнением (<)
bool createShadowMap() {
	auto& context = graphics::internal::context;

	VkImageCreateInfo image_info{};
	image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	image_info.imageType = VK_IMAGE_TYPE_2D;
	image_info.extent = {shadow_map_size, shadow_map_size, 1};
	image_info.mipLevels = 1;
	image_info.arrayLayers = 1;
	image_info.format = VK_FORMAT_D32_SFLOAT;
	image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	image_info.samples = VK_SAMPLE_COUNT_1_BIT;

	VmaAllocationCreateInfo alloc_info{};
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateImage(context.allocator, &image_info, &alloc_info,
					   &shadow_image, &shadow_allocation, nullptr) != VK_SUCCESS) {
		return false;
	}

	VkImageViewCreateInfo view_info{};
	view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	view_info.image = shadow_image;
	view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format = VK_FORMAT_D32_SFLOAT;
	view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	view_info.subresourceRange.levelCount = 1;
	view_info.subresourceRange.layerCount = 1;
	if (vkCreateImageView(context.device, &view_info, nullptr, &shadow_view) != VK_SUCCESS) {
		return false;
	}

	VkSamplerCreateInfo sampler_info{};
	sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	sampler_info.magFilter = VK_FILTER_LINEAR;
	sampler_info.minFilter = VK_FILTER_LINEAR;
	sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
	sampler_info.compareEnable = VK_TRUE;
	sampler_info.compareOp = VK_COMPARE_OP_LESS;
	return vkCreateSampler(context.device, &sampler_info, nullptr, &shadow_sampler) == VK_SUCCESS;
}

bool createDescriptors() {
	auto& device = graphics::internal::context.device;

	VkDescriptorSetLayoutBinding scene_bindings[4]{};
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
	scene_bindings[3].binding = 3;
	scene_bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	scene_bindings[3].descriptorCount = 1;
	scene_bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo scene_info{};
	scene_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	scene_info.bindingCount = 4;
	scene_info.pBindings = scene_bindings;
	if (vkCreateDescriptorSetLayout(device, &scene_info, nullptr, &scene_layout) != VK_SUCCESS) {
		return false;
	}

	VkDescriptorSetLayoutBinding model_bindings[2]{};
	model_bindings[0].binding = 0;
	model_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
	model_bindings[0].descriptorCount = 1;
	model_bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
	model_bindings[1].binding = 1;
	model_bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	model_bindings[1].descriptorCount = 1;
	model_bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

	VkDescriptorSetLayoutCreateInfo model_info{};
	model_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	model_info.bindingCount = 2;
	model_info.pBindings = model_bindings;
	if (vkCreateDescriptorSetLayout(device, &model_info, nullptr, &model_layout) != VK_SUCCESS) {
		return false;
	}

	VkDescriptorPoolSize pool_sizes[4]{};
	pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pool_sizes[0].descriptorCount = 1;
	pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	pool_sizes[1].descriptorCount = 2;
	pool_sizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
	pool_sizes[2].descriptorCount = texture_count;
	pool_sizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	pool_sizes[3].descriptorCount = 1 + texture_count;

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets = 1 + texture_count;
	pool_info.poolSizeCount = 4;
	pool_info.pPoolSizes = pool_sizes;
	if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		return false;
	}

	VkDescriptorSetAllocateInfo scene_alloc{};
	scene_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	scene_alloc.descriptorPool = descriptor_pool;
	scene_alloc.descriptorSetCount = 1;
	scene_alloc.pSetLayouts = &scene_layout;
	if (vkAllocateDescriptorSets(device, &scene_alloc, &scene_set) != VK_SUCCESS) {
		return false;
	}

	VkDescriptorSetLayout material_layouts[texture_count] = {model_layout, model_layout, model_layout};
	VkDescriptorSetAllocateInfo material_alloc{};
	material_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	material_alloc.descriptorPool = descriptor_pool;
	material_alloc.descriptorSetCount = texture_count;
	material_alloc.pSetLayouts = material_layouts;
	if (vkAllocateDescriptorSets(device, &material_alloc, material_sets) != VK_SUCCESS) {
		return false;
	}

	VkDescriptorBufferInfo scene_info_buffer{scene_buffer.buffer, 0, sizeof(SceneData)};
	VkDescriptorBufferInfo point_info{point_buffer.buffer, 0, sizeof(GpuPointLight) * max_lights};
	VkDescriptorBufferInfo spot_info{spot_buffer.buffer, 0, sizeof(GpuSpotLight) * max_lights};
	VkDescriptorBufferInfo model_info_buffer{model_buffer.buffer, 0, sizeof(ModelData)};

	VkDescriptorImageInfo shadow_image_info{};
	shadow_image_info.sampler = shadow_sampler;
	shadow_image_info.imageView = shadow_view;
	shadow_image_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

	VkWriteDescriptorSet scene_writes[4]{};
	scene_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	scene_writes[0].dstSet = scene_set;
	scene_writes[0].dstBinding = 0;
	scene_writes[0].descriptorCount = 1;
	scene_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	scene_writes[0].pBufferInfo = &scene_info_buffer;
	scene_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	scene_writes[1].dstSet = scene_set;
	scene_writes[1].dstBinding = 1;
	scene_writes[1].descriptorCount = 1;
	scene_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	scene_writes[1].pBufferInfo = &point_info;
	scene_writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	scene_writes[2].dstSet = scene_set;
	scene_writes[2].dstBinding = 2;
	scene_writes[2].descriptorCount = 1;
	scene_writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	scene_writes[2].pBufferInfo = &spot_info;
	scene_writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	scene_writes[3].dstSet = scene_set;
	scene_writes[3].dstBinding = 3;
	scene_writes[3].descriptorCount = 1;
	scene_writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	scene_writes[3].pImageInfo = &shadow_image_info;
	vkUpdateDescriptorSets(device, 4, scene_writes, 0, nullptr);

	for (int i = 0; i < texture_count; ++i) {
		VkDescriptorImageInfo image_info{};
		image_info.sampler = texture_sampler;
		image_info.imageView = textures[i].view;
		image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		VkWriteDescriptorSet writes[2]{};
		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet = material_sets[i];
		writes[0].dstBinding = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		writes[0].pBufferInfo = &model_info_buffer;
		writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet = material_sets[i];
		writes[1].dstBinding = 1;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		writes[1].pImageInfo = &image_info;
		vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
	}
	return true;
}

bool createPipelines() {
	auto& device = graphics::internal::context.device;

	VkShaderModule vert = createShaderModule("shaders/hexahedron.vert.spv");
	VkShaderModule frag = createShaderModule("shaders/hexahedron.frag.spv");
	VkShaderModule shadow_vert = createShaderModule("shaders/shadow.vert.spv");
	if (vert == VK_NULL_HANDLE || frag == VK_NULL_HANDLE || shadow_vert == VK_NULL_HANDLE) {
		vkDestroyShaderModule(device, vert, nullptr);
		vkDestroyShaderModule(device, frag, nullptr);
		vkDestroyShaderModule(device, shadow_vert, nullptr);
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
	attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
	attributes[2].offset = offsetof(Vertex, uv);

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
	depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

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

	VkPushConstantRange push_range{};
	push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
	push_range.offset = 0;
	push_range.size = sizeof(PushConstants);

	VkDescriptorSetLayout set_layouts[2] = {scene_layout, model_layout};
	VkPipelineLayoutCreateInfo layout_info{};
	layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout_info.setLayoutCount = 2;
	layout_info.pSetLayouts = set_layouts;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges = &push_range;
	if (vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
		vkDestroyShaderModule(device, vert, nullptr);
		vkDestroyShaderModule(device, frag, nullptr);
		vkDestroyShaderModule(device, shadow_vert, nullptr);
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

	VkResult main_res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);

	// конвейер для карты теней (dynamic rendering, только вершинный шейдер и глубина)
	VkPipelineShaderStageCreateInfo shadow_stage{};
	shadow_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	shadow_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shadow_stage.module = shadow_vert;
	shadow_stage.pName = "main";

	VkPipelineVertexInputStateCreateInfo shadow_vertex_input = vertex_input;
	shadow_vertex_input.vertexAttributeDescriptionCount = 1;

	VkPipelineRasterizationStateCreateInfo shadow_raster = rasterization;
	shadow_raster.depthBiasEnable = VK_TRUE;

	VkDynamicState shadow_dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
		VK_DYNAMIC_STATE_DEPTH_BIAS,
	};
	VkPipelineDynamicStateCreateInfo shadow_dynamic{};
	shadow_dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	shadow_dynamic.dynamicStateCount = 3;
	shadow_dynamic.pDynamicStates = shadow_dynamic_states;

	VkPipelineRenderingCreateInfo rendering_info{};
	rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
	rendering_info.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

	VkGraphicsPipelineCreateInfo shadow_info = pipeline_info;
	shadow_info.pNext = &rendering_info;
	shadow_info.stageCount = 1;
	shadow_info.pStages = &shadow_stage;
	shadow_info.pVertexInputState = &shadow_vertex_input;
	shadow_info.pRasterizationState = &shadow_raster;
	shadow_info.pColorBlendState = nullptr;
	shadow_info.pDynamicState = &shadow_dynamic;
	shadow_info.renderPass = VK_NULL_HANDLE;

	VkResult shadow_res = vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &shadow_info, nullptr, &shadow_pipeline);

	vkDestroyShaderModule(device, vert, nullptr);
	vkDestroyShaderModule(device, frag, nullptr);
	vkDestroyShaderModule(device, shadow_vert, nullptr);
	return main_res == VK_SUCCESS && shadow_res == VK_SUCCESS;
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

glm::mat4 lightViewProjMatrix() {
	glm::vec3 dir = dir_direction;
	if (glm::length(dir) < 0.001f) {
		dir = glm::vec3(0.0f, -1.0f, 0.0f);
	}
	dir = glm::normalize(dir);

	glm::vec3 up(0.0f, 1.0f, 0.0f);
	if (std::abs( glm::dot(dir, up) ) > 0.98f) {
		up = glm::vec3(0.0f, 0.0f, 1.0f);
	}

	glm::mat4 light_view = glm::lookAt(-dir * 10.0f, glm::vec3(0.0f), up);
	glm::mat4 light_proj = glm::ortho(-shadow_area, shadow_area, -shadow_area, shadow_area, 0.5f, 25.0f);
	light_proj[1][1] *= -1.0f;
	return light_proj * light_view;
}

void drawLightList() {
	if (ImGui::CollapsingHeader("Точечные источники")) {
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

	if (ImGui::CollapsingHeader("Прожекторы")) {
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

void drawMesh(VkCommandBuffer cmd, const Buffer& vb, const Buffer& ib, uint32_t count,
			  int tex_index, uint32_t model_index, bool with_push, int effect) {
	VkDeviceSize zero_offset = 0;
	uint32_t dynamic_offset = model_index * uint32_t(model_stride);
	if (with_push) {
		PushConstants push{current_time, effect};
		vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
	}
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
							1, 1, &material_sets[tex_index], 1, &dynamic_offset);
	vkCmdBindVertexBuffers(cmd, 0, 1, &vb.buffer, &zero_offset);
	vkCmdBindIndexBuffer(cmd, ib.buffer, 0, VK_INDEX_TYPE_UINT16);
	vkCmdDrawIndexed(cmd, count, 1, 0, 0, 0);
}

void drawAllObjects(VkCommandBuffer cmd, bool with_push) {
	drawMesh(cmd, room_vertex_buffer, room_index_buffer, room_index_count, room_texture, 1, with_push, 0);
	drawMesh(cmd, vertex_buffer, index_buffer, index_count, cube_texture, 0, with_push, use_effect ? 1 : 0);
}

} // namespace

bool initialize() {
	ImFont* font = ImGui::GetIO().Fonts->AddFontFromFileTTF(
		"C:/Windows/Fonts/segoeui.ttf", 18.0f, nullptr, ImGui::GetIO().Fonts->GetGlyphRangesCyrillic());
	if (font != nullptr) {
		ImGui::GetIO().FontDefault = font;
	}

	auto& context = graphics::internal::context;
	VkPhysicalDeviceProperties props{};
	vkGetPhysicalDeviceProperties(context.physical_device, &props);
	VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
	model_stride = (sizeof(ModelData) + align - 1) & ~(align - 1);

	VkCommandPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
	pool_info.queueFamilyIndex = context.graphics_queue_index;
	if (vkCreateCommandPool(context.device, &pool_info, nullptr, &transfer_pool) != VK_SUCCESS) {
		return false;
	}

	scene_buffer = createBuffer(sizeof(SceneData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	model_buffer = createBuffer(model_stride * model_count, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	point_buffer = createBuffer(sizeof(GpuPointLight) * max_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
	spot_buffer = createBuffer(sizeof(GpuSpotLight) * max_lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	if (!createCube() || !createRoom() || !createTextures() || !createShadowMap() ||
		scene_buffer.data == nullptr || model_buffer.data == nullptr ||
		point_buffer.data == nullptr || spot_buffer.data == nullptr ||
		!createDescriptors() || !createPipelines()) {
		shutdown();
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vkDestroyPipeline(context.device, shadow_pipeline, nullptr);
	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, model_layout, nullptr);
	vkDestroyDescriptorSetLayout(context.device, scene_layout, nullptr);
	shadow_pipeline = VK_NULL_HANDLE;
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	model_layout = VK_NULL_HANDLE;
	scene_layout = VK_NULL_HANDLE;

	vkDestroySampler(context.device, shadow_sampler, nullptr);
	vkDestroyImageView(context.device, shadow_view, nullptr);
	if (shadow_image != VK_NULL_HANDLE) {
		vmaDestroyImage(context.allocator, shadow_image, shadow_allocation);
	}
	shadow_sampler = VK_NULL_HANDLE;
	shadow_view = VK_NULL_HANDLE;
	shadow_image = VK_NULL_HANDLE;

	for (Texture& tex : textures) {
		vkDestroyImageView(context.device, tex.view, nullptr);
		if (tex.image != VK_NULL_HANDLE) {
			vmaDestroyImage(context.allocator, tex.image, tex.allocation);
		}
		tex = {};
	}
	vkDestroySampler(context.device, texture_sampler, nullptr);
	texture_sampler = VK_NULL_HANDLE;

	vkDestroyCommandPool(context.device, transfer_pool, nullptr);
	transfer_pool = VK_NULL_HANDLE;

	destroyBuffer(spot_buffer);
	destroyBuffer(point_buffer);
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
	current_time = float(time);
	if (playing) {
		anim_time += dt * speed;
	}

	ImGui::SetNextWindowSize(ImVec2(430.0f, 660.0f), ImGuiCond_FirstUseEver);
	ImGui::Begin("Лабораторная 4");
	ImGui::TextUnformatted("Вариант 12");

	if (ImGui::CollapsingHeader("Карта теней", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat("Площадь ортопроекции", &shadow_area, 2.0f, 16.0f);
		ImGui::SliderFloat("Depth bias (const)", &depth_bias_constant, 0.0f, 8.0f);
		ImGui::SliderFloat("Depth bias (slope)", &depth_bias_slope, 0.0f, 8.0f);
	}

	if (ImGui::CollapsingHeader("Текстуры", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Combo("Текстура куба", &cube_texture, texture_names, texture_count);
		ImGui::Combo("Текстура комнаты", &room_texture, texture_names, texture_count);
		ImGui::Checkbox("Эффект сэмплирования на кубе", &use_effect);
	}

	if (ImGui::CollapsingHeader("Направленный свет (солнце)", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Фоновый свет", &ambient.x);
		ImGui::DragFloat3("Направление лучей", &dir_direction.x, 0.02f);
		ImGui::ColorEdit3("Цвет света", &dir_color.x);
		ImGui::SliderFloat("Интенсивность", &dir_intensity, 0.0f, 5.0f);
	}

	if (ImGui::CollapsingHeader("Фигура", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Позиция", &position.x, 0.02f);
		ImGui::DragFloat3("Поворот", &rotation.x, 0.5f);
		ImGui::DragFloat3("Масштаб", &scale.x, 0.01f, 0.05f, 5.0f);
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

	if (ImGui::CollapsingHeader("Проекция")) {
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

	if (ImGui::CollapsingHeader("Материал")) {
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
	scene.light_view_proj = lightViewProjMatrix();
	scene.camera_position = glm::vec4(camera_position, 1.0f);
	scene.ambient = glm::vec4(ambient, 1.0f);
	scene.dir_direction = glm::vec4(dir_direction, 0.0f);
	scene.dir_color = glm::vec4(dir_color, dir_intensity);
	scene.counts = glm::ivec4(int(point_lights.size()), int(spot_lights.size()), 0, 0);
	std::memcpy(scene_buffer.data, &scene, sizeof(scene));

	auto* base = static_cast<char*>(model_buffer.data);
	auto writeModel = [&](int index, const glm::mat4& mat, glm::vec3 col, glm::vec4 spec) {
		ModelData data;
		data.model = mat;
		data.albedo = glm::vec4(col, 1.0f);
		data.specular = spec;
		std::memcpy(base + model_stride * index, &data, sizeof(data));
	};

	writeModel(0, modelMatrix(), albedo, glm::vec4(specular, shininess));
	writeModel(1, glm::mat4(1.0f), glm::vec3(0.55f, 0.55f, 0.58f), glm::vec4(0.08f, 0.08f, 0.08f, 16.0f));

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
	vmaFlushAllocation(context.allocator, point_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(context.allocator, spot_buffer.allocation, 0, VK_WHOLE_SIZE);

	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo begin{};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &begin);

	// 1. проход карты теней через dynamic rendering
	VkImageMemoryBarrier to_depth{};
	to_depth.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	to_depth.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	to_depth.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	to_depth.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_depth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	to_depth.image = shadow_image;
	to_depth.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	to_depth.subresourceRange.levelCount = 1;
	to_depth.subresourceRange.layerCount = 1;
	to_depth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
						 0, 0, nullptr, 0, nullptr, 1, &to_depth);

	VkRenderingAttachmentInfo depth_attachment{};
	depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
	depth_attachment.imageView = shadow_view;
	depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depth_attachment.clearValue.depthStencil = {1.0f, 0};

	VkRenderingInfo rendering_info{};
	rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
	rendering_info.renderArea.extent = {shadow_map_size, shadow_map_size};
	rendering_info.layerCount = 1;
	rendering_info.pDepthAttachment = &depth_attachment;

	vkCmdBeginRendering(cmd, &rendering_info);

	VkViewport shadow_viewport{};
	shadow_viewport.width = float(shadow_map_size);
	shadow_viewport.height = float(shadow_map_size);
	shadow_viewport.maxDepth = 1.0f;
	vkCmdSetViewport(cmd, 0, 1, &shadow_viewport);

	VkRect2D shadow_scissor{};
	shadow_scissor.extent = {shadow_map_size, shadow_map_size};
	vkCmdSetScissor(cmd, 0, 1, &shadow_scissor);
	vkCmdSetDepthBias(cmd, depth_bias_constant, 0.0f, depth_bias_slope);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &scene_set, 0, nullptr);
	drawAllObjects(cmd, false);

	vkCmdEndRendering(cmd);

	VkImageMemoryBarrier to_read = to_depth;
	to_read.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
	to_read.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
	to_read.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	to_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
						 0, 0, nullptr, 0, nullptr, 1, &to_read);

	// 2. основной проход сцены
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

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &scene_set, 0, nullptr);
	drawAllObjects(cmd, true);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
