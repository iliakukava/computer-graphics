#include "application.hpp"

#include <cmath>
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

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

struct Uniforms {
	glm::mat4 mvp;
	glm::vec4 tint;
};

struct Buffer {
	VkBuffer buffer = VK_NULL_HANDLE;
	VmaAllocation allocation = VK_NULL_HANDLE;
	void* data = nullptr;
};

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

Buffer vertex_buffer;
Buffer index_buffer;
Buffer uniform_buffer;
uint32_t index_count = 0;

VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
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
	Vertex vertices[] = {
		{{-1.0f, -1.0f, -1.0f}},
		{{ 1.0f, -1.0f, -1.0f}},
		{{ 1.0f,  1.0f, -1.0f}},
		{{-1.0f,  1.0f, -1.0f}},
		{{-1.0f, -1.0f,  1.0f}},
		{{ 1.0f, -1.0f,  1.0f}},
		{{ 1.0f,  1.0f,  1.0f}},
		{{-1.0f,  1.0f,  1.0f}},
	};

	// цвет вершины зависит от её позиции в локальных координатах
	for (Vertex& vertex : vertices) {
		vertex.color = vertex.position * 0.5f + glm::vec3(0.5f);
	}

	const uint16_t indices[] = {
		4, 5, 6, 4, 6, 7,
		1, 0, 3, 1, 3, 2,
		0, 4, 7, 0, 7, 3,
		5, 1, 2, 5, 2, 6,
		0, 1, 5, 0, 5, 4,
		3, 7, 6, 3, 6, 2,
	};
	index_count = uint32_t(sizeof(indices) / sizeof(indices[0]));

	vertex_buffer = createBuffer(sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	index_buffer = createBuffer(sizeof(indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
	if (vertex_buffer.data == nullptr || index_buffer.data == nullptr) {
		return false;
	}

	std::memcpy(vertex_buffer.data, vertices, sizeof(vertices));
	std::memcpy(index_buffer.data, indices, sizeof(indices));
	vmaFlushAllocation(graphics::internal::context.allocator, vertex_buffer.allocation, 0, VK_WHOLE_SIZE);
	vmaFlushAllocation(graphics::internal::context.allocator, index_buffer.allocation, 0, VK_WHOLE_SIZE);
	return true;
}

bool createDescriptors() {
	auto& device = graphics::internal::context.device;

	VkDescriptorSetLayoutBinding binding{};
	binding.binding = 0;
	binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	binding.descriptorCount = 1;
	binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

	VkDescriptorSetLayoutCreateInfo layout_info{};
	layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layout_info.bindingCount = 1;
	layout_info.pBindings = &binding;
	if (vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	VkDescriptorPoolSize pool_size{};
	pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	pool_size.descriptorCount = 1;

	VkDescriptorPoolCreateInfo pool_info{};
	pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	pool_info.maxSets = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes = &pool_size;
	if (vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	VkDescriptorSetAllocateInfo alloc_info{};
	alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	alloc_info.descriptorPool = descriptor_pool;
	alloc_info.descriptorSetCount = 1;
	alloc_info.pSetLayouts = &descriptor_set_layout;
	if (vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	VkDescriptorBufferInfo buffer_info{};
	buffer_info.buffer = uniform_buffer.buffer;
	buffer_info.range = sizeof(Uniforms);

	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.dstSet = descriptor_set;
	write.dstBinding = 0;
	write.descriptorCount = 1;
	write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	write.pBufferInfo = &buffer_info;
	vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
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

	VkVertexInputAttributeDescription attributes[2]{};
	attributes[0].location = 0;
	attributes[0].binding = 0;
	attributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[0].offset = offsetof(Vertex, position);
	attributes[1].location = 1;
	attributes[1].binding = 0;
	attributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
	attributes[1].offset = offsetof(Vertex, color);

	VkPipelineVertexInputStateCreateInfo vertex_input{};
	vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
	vertex_input.vertexBindingDescriptionCount = 1;
	vertex_input.pVertexBindingDescriptions = &binding;
	vertex_input.vertexAttributeDescriptionCount = 2;
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

	VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	VkPipelineDynamicStateCreateInfo dynamic{};
	dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates = dynamic_states;

	VkPipelineLayoutCreateInfo layout_info{};
	layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	layout_info.setLayoutCount = 1;
	layout_info.pSetLayouts = &descriptor_set_layout;
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
	// восьмёрка в XZ и покачивание по Y
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
	return model;
}

glm::mat4 projectionMatrix(float aspect) {
	glm::mat4 projection;
	if (perspective) {
		projection = glm::perspective(glm::radians(fov), aspect, 0.1f, 100.0f);
	} else {
		float half_width = ortho_size * aspect;
		projection = glm::ortho(-half_width, half_width, -ortho_size, ortho_size, 0.1f, 100.0f);
	}

	// в Vulkan ось Y смотрит вниз
	projection[1][1] *= -1.0f;
	return projection;
}

} // namespace

bool initialize() {
	ImFont* font = ImGui::GetIO().Fonts->AddFontFromFileTTF(
		"C:/Windows/Fonts/segoeui.ttf", 18.0f, nullptr, ImGui::GetIO().Fonts->GetGlyphRangesCyrillic());
	if (font != nullptr) {
		ImGui::GetIO().FontDefault = font;
	}

	uniform_buffer = createBuffer(sizeof(Uniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
	if (!createCube() || uniform_buffer.data == nullptr || !createDescriptors() || !createPipeline()) {
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
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	descriptor_pool = VK_NULL_HANDLE;
	descriptor_set_layout = VK_NULL_HANDLE;

	destroyBuffer(uniform_buffer);
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

	ImGui::Begin("Правильный гексаэдр");
	ImGui::TextUnformatted("Вариант 12");

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

	ImGui::DragFloat3("Позиция", &position.x, 0.02f);
	ImGui::DragFloat3("Поворот", &rotation.x, 0.5f);
	ImGui::DragFloat3("Масштаб", &scale.x, 0.01f, 0.05f, 5.0f);
	ImGui::ColorEdit3("Цвет", &tint.x);

	if (ImGui::Button(playing ? "Пауза" : "Воспроизвести")) {
		playing = !playing;
	}
	ImGui::SliderFloat("Скорость", &speed, 0.0f, 4.0f);
	ImGui::SliderFloat("Радиус", &radius, 0.0f, 3.0f);
	ImGui::SliderFloat("Высота", &height, 0.0f, 2.0f);
	ImGui::SliderFloat("Скорость вращения", &rotation_speed, 0.0f, 4.0f);
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	VkCommandBuffer cmd = fd.command_buffer;
	VkExtent2D extent = context.swapchain_extent;

	vkResetCommandBuffer(cmd, 0);

	VkCommandBufferBeginInfo begin{};
	begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	vkBeginCommandBuffer(cmd, &begin);

	float aspect = float(extent.width) / float(extent.height);
	glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 2.2f, 8.0f),
								 glm::vec3(0.0f),
								 glm::vec3(0.0f, 1.0f, 0.0f));

	Uniforms uniforms;
	uniforms.mvp = projectionMatrix(aspect) * view * modelMatrix();
	uniforms.tint = glm::vec4(tint, 1.0f);
	std::memcpy(uniform_buffer.data, &uniforms, sizeof(uniforms));
	vmaFlushAllocation(context.allocator, uniform_buffer.allocation, 0, VK_WHOLE_SIZE);

	VkClearValue clears[2]{};
	clears[0].color.float32[0] = 0.16f;
	clears[0].color.float32[1] = 0.18f;
	clears[0].color.float32[2] = 0.22f;
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

	VkDeviceSize offset = 0;
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer.buffer, 0, VK_INDEX_TYPE_UINT16);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
	vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
