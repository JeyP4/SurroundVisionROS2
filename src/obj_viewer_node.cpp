#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <limits>
#include <sstream>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "rclcpp/executors/single_threaded_executor.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp/node.hpp"
#include "rclcpp/utilities.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

// Shader sources (simple Phong lighting)
const char* vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;

uniform sampler2D texture_diffuse;
uniform vec3 viewPos;
uniform vec3 lightDirection;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform float specularStrength;
uniform float shininess;

void main() {
    vec3 norm = normalize(Normal);
    vec3 dir = normalize(-lightDirection);

    vec3 diffuse = max(dot(norm, dir), 0.0) * lightColor;

    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-dir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 texColor = texture(texture_diffuse, TexCoord).rgb;
    vec3 result = (ambientColor + diffuse + specular) * texColor;
    FragColor = vec4(result, 1.0);
}
)glsl";

const char* axisVertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vColor = aColor;
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)glsl";

const char* axisFragmentShaderSource = R"glsl(
#version 330 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
)glsl";

class ObjViewerNode : public rclcpp::Node {
public:
    ObjViewerNode() : Node("obj_viewer_node") {
        package_share_directory_ = ament_index_cpp::get_package_share_directory("surround_vision");
        declare_and_load_parameters();
        axis_alignment_ = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        update_camera_vectors();
    }

    void run() {
        if (!init_window()) {
            RCLCPP_ERROR(this->get_logger(), "Window initialisation failed.");
            cleanup();
            return;
        }
        if (!load_model()) {
            RCLCPP_ERROR(this->get_logger(), "Model load failed.");
            cleanup();
            return;
        }
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(this->shared_from_this());
        main_loop(executor);
        executor.remove_node(this->shared_from_this());
        cleanup();
    }

private:
    struct CameraState {
        glm::vec3 target {0.0f, 0.0f, 0.0f};
        float distance {6.0f};
        float yaw {-90.0f};
        float pitch {-15.0f};
        glm::vec3 position {0.0f, 0.0f, 5.0f};
        glm::vec3 front {0.0f, 0.0f, -1.0f};
        glm::vec3 up {0.0f, 1.0f, 0.0f};
        glm::vec3 right {1.0f, 0.0f, 0.0f};
    };

    struct Mesh {
        GLuint vao = 0;
        GLuint vbo = 0;
        GLuint ebo = 0;
        GLsizei index_count = 0;
        GLuint texture_id = 0;
    };

    static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
        if (auto* self = static_cast<ObjViewerNode*>(glfwGetWindowUserPointer(window))) {
            self->handle_framebuffer_size(width, height);
        }
    }

    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
        if (auto* self = static_cast<ObjViewerNode*>(glfwGetWindowUserPointer(window))) {
            self->handle_cursor_position(xpos, ypos);
        }
    }

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        (void)xoffset;
        if (auto* self = static_cast<ObjViewerNode*>(glfwGetWindowUserPointer(window))) {
            self->handle_scroll(yoffset);
        }
    }

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (auto* self = static_cast<ObjViewerNode*>(glfwGetWindowUserPointer(window))) {
            self->handle_mouse_button(window, button, action, mods);
        }
    }

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        (void)scancode;
        if (auto* self = static_cast<ObjViewerNode*>(glfwGetWindowUserPointer(window))) {
            self->handle_key(key, action, mods);
        }
    }

    bool init_window() {
        if (!glfwInit()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize GLFW");
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        window_ = glfwCreateWindow(window_width_, window_height_, window_title_with_controls_.c_str(), nullptr, nullptr);
        if (!window_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create GLFW window");
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetCursorPosCallback(window_, cursor_position_callback);
        glfwSetScrollCallback(window_, scroll_callback);
        glfwSetMouseButtonCallback(window_, mouse_button_callback);
        glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
        glfwSetKeyCallback(window_, key_callback);
        glfwSwapInterval(1);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize GLEW");
            return false;
        }

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glFrontFace(GL_CCW);
        glViewport(0, 0, window_width_, window_height_);
        
        setup_shaders();
        create_default_texture();
        setup_axis_renderer();
        return true;
    }

    void setup_shaders() {
        // Vertex shader
        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
        glCompileShader(vertexShader);
        // Check for compile errors
        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Vertex shader compilation failed: %s", infoLog);
        }

        // Fragment shader
        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
        glCompileShader(fragmentShader);
        // Check for compile errors
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Fragment shader compilation failed: %s", infoLog);
        }

        // Link shaders
        shader_program_ = glCreateProgram();
        glAttachShader(shader_program_, vertexShader);
        glAttachShader(shader_program_, fragmentShader);
        glLinkProgram(shader_program_);
        // Check for linking errors
        glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader_program_, 512, NULL, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Shader program linking failed: %s", infoLog);
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
    }

    void setup_axis_renderer() {
        if (!show_axes_) {
            return;
        }

        GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &axisVertexShaderSource, nullptr);
        glCompileShader(vertexShader);

        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Axis vertex shader compilation failed: %s", infoLog);
        }

        GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &axisFragmentShaderSource, nullptr);
        glCompileShader(fragmentShader);
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Axis fragment shader compilation failed: %s", infoLog);
        }

        axis_program_ = glCreateProgram();
        glAttachShader(axis_program_, vertexShader);
        glAttachShader(axis_program_, fragmentShader);
        glLinkProgram(axis_program_);
        glGetProgramiv(axis_program_, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(axis_program_, 512, nullptr, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Axis shader program linking failed: %s", infoLog);
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        std::vector<float> axis_vertices = {
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            axis_length_, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,

            0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, axis_length_, 0.0f, 0.0f, 1.0f, 0.0f,

            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, axis_length_, 0.0f, 0.0f, 1.0f
        };

        glGenVertexArrays(1, &axis_vao_);
        glGenBuffers(1, &axis_vbo_);

        glBindVertexArray(axis_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, axis_vbo_);
        glBufferData(GL_ARRAY_BUFFER, axis_vertices.size() * sizeof(float), axis_vertices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }
    
    GLuint load_texture(const std::string& path) {
        GLuint textureID;
        glGenTextures(1, &textureID);

        int width, height, nrComponents;
        stbi_set_flip_vertically_on_load(true);
        unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
        if (data) {
            GLenum format;
            if (nrComponents == 1) format = GL_RED;
            else if (nrComponents == 3) format = GL_RGB;
            else if (nrComponents == 4) format = GL_RGBA;
            else {
                 RCLCPP_ERROR(this->get_logger(), "Unsupported texture format with %d components", nrComponents);
                 stbi_image_free(data);
                 glDeleteTextures(1, &textureID);
                 return 0;
            }

            glBindTexture(GL_TEXTURE_2D, textureID);
            glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Texture failed to load at path: %s", path.c_str());
            glDeleteTextures(1, &textureID);
            textureID = 0;
        }
        stbi_image_free(data);

        return textureID;
    }


    bool load_model() {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        std::string model_path = resolved_model_path_;
        std::string mtl_dir = model_directory_;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, model_path.c_str(), mtl_dir.c_str())) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s%s", warn.c_str(), err.c_str());
            return false;
        }

        if (!warn.empty()) {
            RCLCPP_WARN(this->get_logger(), "OBJ Loader Warning: %s", warn.c_str());
        }

        // Load textures
        std::vector<GLuint> texture_ids;
        for(const auto& material : materials) {
            if (!material.diffuse_texname.empty()) {
                texture_ids.push_back(load_texture(mtl_dir + material.diffuse_texname));
            } else {
                texture_ids.push_back(0); // No texture
            }
        }

        material_textures_ = texture_ids;

        for (const auto& shape : shapes) {
            std::vector<float> vertices;
            std::vector<unsigned int> indices;
            
            // Assume one material per shape for simplicity
            int material_id = -1;
            if(!shape.mesh.material_ids.empty()){
                material_id = shape.mesh.material_ids[0];
            }


            for (const auto& index : shape.mesh.indices) {
                // Vertex
                vertices.push_back(attrib.vertices[3 * index.vertex_index + 0]);
                vertices.push_back(attrib.vertices[3 * index.vertex_index + 1]);
                vertices.push_back(attrib.vertices[3 * index.vertex_index + 2]);
                // Normal
                if (index.normal_index >= 0) {
                    vertices.push_back(attrib.normals[3 * index.normal_index + 0]);
                    vertices.push_back(attrib.normals[3 * index.normal_index + 1]);
                    vertices.push_back(attrib.normals[3 * index.normal_index + 2]);
                } else {
                    vertices.push_back(0.0f); vertices.push_back(0.0f); vertices.push_back(0.0f);
                }
                // Texture Coordinate
                if (index.texcoord_index >= 0) {
                    vertices.push_back(attrib.texcoords[2 * index.texcoord_index + 0]);
                    vertices.push_back(attrib.texcoords[2 * index.texcoord_index + 1]);
                } else {
                    vertices.push_back(0.0f); vertices.push_back(0.0f);
                }
                indices.push_back(indices.size());
            }

            Mesh mesh;
            mesh.index_count = indices.size();
            if (material_id != -1 && static_cast<size_t>(material_id) < texture_ids.size()) {
                 mesh.texture_id = texture_ids[material_id] == 0 ? default_texture_id_ : texture_ids[material_id];
            } else {
                 mesh.texture_id = default_texture_id_;
            }

            glGenVertexArrays(1, &mesh.vao);
            glGenBuffers(1, &mesh.vbo);
            glGenBuffers(1, &mesh.ebo);

            glBindVertexArray(mesh.vao);

            glBindBuffer(GL_ARRAY_BUFFER, mesh.vbo);
            glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

            // Position attribute
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            // Normal attribute
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);
            // Texture coordinate attribute
            glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
            glEnableVertexAttribArray(2);

            glBindVertexArray(0);
            meshes_.push_back(mesh);
        }
        return true;
    }

    void main_loop(rclcpp::executors::SingleThreadedExecutor & executor) {
        while (rclcpp::ok() && !glfwWindowShouldClose(window_)) {
            executor.spin_some();

            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glViewport(0, 0, viewport_width_, viewport_height_);

            glUseProgram(shader_program_);
            apply_lighting_uniforms();

            // Camera/View transformation
            glm::mat4 view = glm::lookAt(camera_state_.position, camera_state_.target, camera_state_.up);
            // Projection transformation
            float aspect_ratio = viewport_height_ > 0 ? static_cast<float>(viewport_width_) / static_cast<float>(viewport_height_) : 1.0f;
            glm::mat4 projection = glm::perspective(glm::radians(field_of_view_deg_), aspect_ratio, 0.1f, 100.0f);

            glUniformMatrix4fv(glGetUniformLocation(shader_program_, "view"), 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(shader_program_, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
            glUniform3fv(glGetUniformLocation(shader_program_, "viewPos"), 1, glm::value_ptr(camera_state_.position));

            // Model transformation
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::scale(model, glm::vec3(model_scale_));
            glUniformMatrix4fv(glGetUniformLocation(shader_program_, "model"), 1, GL_FALSE, glm::value_ptr(model));

            for (const auto& mesh : meshes_) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mesh.texture_id);
                glUniform1i(glGetUniformLocation(shader_program_, "texture_diffuse"), 0);

                glBindVertexArray(mesh.vao);
                glDrawElements(GL_TRIANGLES, mesh.index_count, GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            }

            render_axes(view, projection, model);

            glfwSwapBuffers(window_);
            glfwPollEvents();
        }
    }

    void render_axes(const glm::mat4& view, const glm::mat4& projection, const glm::mat4& model) {
        if (!show_axes_ || axis_program_ == 0 || axis_vao_ == 0) {
            return;
        }

        glUseProgram(axis_program_);
        glUniformMatrix4fv(glGetUniformLocation(axis_program_, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(axis_program_, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glm::mat4 axis_model = model * axis_alignment_;
        glUniformMatrix4fv(glGetUniformLocation(axis_program_, "model"), 1, GL_FALSE, glm::value_ptr(axis_model));

        glBindVertexArray(axis_vao_);
        glLineWidth(axis_line_width_);
        glDrawArrays(GL_LINES, 0, 6);
        glBindVertexArray(0);
    }

    void cleanup() {
        for(auto& mesh : meshes_){
            glDeleteVertexArrays(1, &mesh.vao);
            glDeleteBuffers(1, &mesh.vbo);
            glDeleteBuffers(1, &mesh.ebo);
        }
        meshes_.clear();
        for (auto texture_id : material_textures_) {
            if (texture_id != 0 && texture_id != default_texture_id_) {
                glDeleteTextures(1, &texture_id);
            }
        }
        material_textures_.clear();
        if (default_texture_id_ != 0) {
            glDeleteTextures(1, &default_texture_id_);
            default_texture_id_ = 0;
        }
        if (shader_program_ != 0) {
            glDeleteProgram(shader_program_);
            shader_program_ = 0;
        }
        if (axis_vbo_ != 0) {
            glDeleteBuffers(1, &axis_vbo_);
            axis_vbo_ = 0;
        }
        if (axis_vao_ != 0) {
            glDeleteVertexArrays(1, &axis_vao_);
            axis_vao_ = 0;
        }
        if (axis_program_ != 0) {
            glDeleteProgram(axis_program_);
            axis_program_ = 0;
        }
        if (window_ != nullptr) {
            glfwDestroyWindow(window_);
            window_ = nullptr;
        }
        glfwTerminate();
    }

    void declare_and_load_parameters() {
        window_width_ = this->declare_parameter<int>("window_width", 1280);
        window_height_ = this->declare_parameter<int>("window_height", 720);
        viewport_width_ = window_width_;
        viewport_height_ = window_height_;

        window_title_ = this->declare_parameter<std::string>("window_title", "ROS 2 OBJ Viewer");
        window_title_with_controls_ = build_window_title_with_controls(window_title_);

        std::string model_param = this->declare_parameter<std::string>("model_path", "models/smartCar/smartCar.obj");
        resolved_model_path_ = resolve_model_path(model_param);
        model_directory_ = std::filesystem::path(resolved_model_path_).parent_path().string() + "/";

        model_scale_ = static_cast<float>(this->declare_parameter<double>("model_uniform_scale", 1.0));
        field_of_view_deg_ = static_cast<float>(this->declare_parameter<double>("field_of_view_deg", 45.0));

        auto camera_target_param = this->declare_parameter<std::vector<double>>("camera_target", {0.0, 0.5, 0.0});
        camera_target_initial_ = parse_vec3_parameter("camera_target", camera_target_param, glm::vec3(0.0f, 0.5f, 0.0f));
        camera_state_.target = camera_target_initial_;

        camera_distance_initial_ = static_cast<float>(this->declare_parameter<double>("camera_distance", 6.0));
        camera_yaw_initial_ = static_cast<float>(this->declare_parameter<double>("camera_yaw_deg", -120.0));
        camera_pitch_initial_ = static_cast<float>(this->declare_parameter<double>("camera_pitch_deg", -20.0));

        camera_state_.distance = camera_distance_initial_;
        camera_state_.yaw = camera_yaw_initial_;
        camera_state_.pitch = camera_pitch_initial_;

        min_camera_distance_ = static_cast<float>(this->declare_parameter<double>("camera_min_distance", 1.0));
        max_camera_distance_ = static_cast<float>(this->declare_parameter<double>("camera_max_distance", 50.0));

        mouse_rotate_sensitivity_ = static_cast<float>(this->declare_parameter<double>("mouse_rotate_sensitivity", 0.2));
        mouse_pan_sensitivity_ = static_cast<float>(this->declare_parameter<double>("mouse_pan_sensitivity", 0.002));
        scroll_sensitivity_ = static_cast<float>(this->declare_parameter<double>("scroll_zoom_sensitivity", 0.8));

        auto ambient_param = this->declare_parameter<std::vector<double>>("ambient_color", {0.2, 0.2, 0.2});
        ambient_color_ = parse_vec3_parameter("ambient_color", ambient_param, glm::vec3(0.2f, 0.2f, 0.2f));

        auto light_color_param = this->declare_parameter<std::vector<double>>("light_color", {1.0, 1.0, 1.0});
        light_color_ = parse_vec3_parameter("light_color", light_color_param, glm::vec3(1.0f, 1.0f, 1.0f));

        auto light_dir_param = this->declare_parameter<std::vector<double>>("light_direction", {-0.5, -1.0, -0.3});
        light_direction_ = parse_vec3_parameter("light_direction", light_dir_param, glm::vec3(-0.5f, -1.0f, -0.3f));
        if (glm::length(light_direction_) < std::numeric_limits<float>::epsilon()) {
            RCLCPP_WARN(this->get_logger(), "Light direction vector is near zero; falling back to default.");
            light_direction_ = glm::vec3(-0.5f, -1.0f, -0.3f);
        }
        light_direction_ = glm::normalize(light_direction_);

        specular_strength_ = static_cast<float>(this->declare_parameter<double>("specular_strength", 0.6));
        shininess_ = static_cast<float>(this->declare_parameter<double>("shininess", 32.0));

        show_axes_ = this->declare_parameter<bool>("show_axes", true);
        axis_length_ = static_cast<float>(this->declare_parameter<double>("axis_length", 2.0));
        axis_length_ = std::max(axis_length_, 0.1f);
    }

    glm::vec3 parse_vec3_parameter(const std::string& name,
                                   const std::vector<double>& values,
                                   const glm::vec3& default_value) {
        if (values.size() != 3) {
            RCLCPP_WARN(this->get_logger(), "Parameter '%s' must contain exactly 3 elements. Using default.", name.c_str());
            return default_value;
        }
        return glm::vec3(static_cast<float>(values[0]),
                         static_cast<float>(values[1]),
                         static_cast<float>(values[2]));
    }

    std::string resolve_model_path(const std::string& model_param) const {
        std::filesystem::path path(model_param);
        if (path.is_absolute()) {
            return path.lexically_normal().string();
        }
        auto combined = std::filesystem::path(package_share_directory_) / path;
        return combined.lexically_normal().string();
    }

    std::string build_window_title_with_controls(const std::string& base_title) const {
        std::ostringstream oss;
        oss << base_title
            << " | Controls: Left drag=Orbit, Right drag=Pan, Scroll=Zoom, R=Reset";
        return oss.str();
    }

    void update_camera_vectors() {
        const float pitch_clamped = glm::clamp(camera_state_.pitch, -89.0f, 89.0f);
        float yaw_rad = glm::radians(camera_state_.yaw);
        float pitch_rad = glm::radians(pitch_clamped);

        glm::vec3 offset;
        offset.x = camera_state_.distance * cosf(pitch_rad) * cosf(yaw_rad);
        offset.y = camera_state_.distance * sinf(pitch_rad);
        offset.z = camera_state_.distance * cosf(pitch_rad) * sinf(yaw_rad);

        camera_state_.position = camera_state_.target + offset;
        camera_state_.front = glm::normalize(camera_state_.target - camera_state_.position);
        camera_state_.right = glm::normalize(glm::cross(camera_state_.front, world_up_));
        camera_state_.up = glm::normalize(glm::cross(camera_state_.right, camera_state_.front));
    }

    void handle_cursor_position(double xpos, double ypos) {
        if (!tracking_mouse_) {
            last_x_ = xpos;
            last_y_ = ypos;
            return;
        }

        double xoffset = xpos - last_x_;
        double yoffset = ypos - last_y_;
        last_x_ = xpos;
        last_y_ = ypos;

        if (rotating_) {
            camera_state_.yaw += static_cast<float>(xoffset) * mouse_rotate_sensitivity_;
            camera_state_.pitch -= static_cast<float>(yoffset) * mouse_rotate_sensitivity_;
            camera_state_.pitch = glm::clamp(camera_state_.pitch, -89.0f, 89.0f);
            update_camera_vectors();
        } else if (panning_) {
            float pan_scale = camera_state_.distance * mouse_pan_sensitivity_;
            camera_state_.target -= camera_state_.right * static_cast<float>(xoffset) * pan_scale;
            camera_state_.target -= camera_state_.up * static_cast<float>(yoffset) * pan_scale;
            update_camera_vectors();
        }
    }

    void handle_scroll(double yoffset) {
        camera_state_.distance -= static_cast<float>(yoffset) * scroll_sensitivity_;
        camera_state_.distance = glm::clamp(camera_state_.distance, min_camera_distance_, max_camera_distance_);
        update_camera_vectors();
    }

    void handle_mouse_button(GLFWwindow* window, int button, int action, int mods) {
        (void)mods;
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            rotating_ = (action == GLFW_PRESS);
        } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            panning_ = (action == GLFW_PRESS);
        }

        if (action == GLFW_PRESS) {
            tracking_mouse_ = true;
            double xpos = 0.0;
            double ypos = 0.0;
            glfwGetCursorPos(window, &xpos, &ypos);
            last_x_ = xpos;
            last_y_ = ypos;
        } else if (!rotating_ && !panning_) {
            tracking_mouse_ = false;
        }
    }

    void handle_key(int key, int action, int mods) {
        if (key == GLFW_KEY_R && action == GLFW_PRESS) {
            (void)mods;
            reset_camera();
        }
    }

    void handle_framebuffer_size(int width, int height) {
        viewport_width_ = std::max(width, 1);
        viewport_height_ = std::max(height, 1);
        glViewport(0, 0, viewport_width_, viewport_height_);
    }

    void reset_camera() {
        camera_state_.target = camera_target_initial_;
        camera_state_.distance = camera_distance_initial_;
        camera_state_.yaw = camera_yaw_initial_;
        camera_state_.pitch = camera_pitch_initial_;
        update_camera_vectors();
    }

    void create_default_texture() {
        if (default_texture_id_ != 0) {
            return;
        }
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        unsigned char white_pixel[4] = {255, 255, 255, 255};
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, white_pixel);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        default_texture_id_ = textureID;
    }

    void apply_lighting_uniforms() {
        glUniform3fv(glGetUniformLocation(shader_program_, "lightDirection"), 1, glm::value_ptr(light_direction_));
        glUniform3fv(glGetUniformLocation(shader_program_, "lightColor"), 1, glm::value_ptr(light_color_));
        glUniform3fv(glGetUniformLocation(shader_program_, "ambientColor"), 1, glm::value_ptr(ambient_color_));
        glUniform1f(glGetUniformLocation(shader_program_, "specularStrength"), specular_strength_);
        glUniform1f(glGetUniformLocation(shader_program_, "shininess"), shininess_);
    }

    GLFWwindow* window_ = nullptr;
    GLuint shader_program_ = 0;
    std::vector<Mesh> meshes_;
    std::string package_share_directory_;
    std::vector<GLuint> material_textures_;
    GLuint default_texture_id_ = 0;
    GLuint axis_program_ = 0;
    GLuint axis_vao_ = 0;
    GLuint axis_vbo_ = 0;

    std::string resolved_model_path_;
    std::string model_directory_;
    int window_width_ = 1280;
    int window_height_ = 720;
    int viewport_width_ = 1280;
    int viewport_height_ = 720;
    std::string window_title_ = "ROS 2 OBJ Viewer";
    std::string window_title_with_controls_;
    float model_scale_ = 1.0f;
    float field_of_view_deg_ = 45.0f;

    CameraState camera_state_;
    glm::vec3 camera_target_initial_{0.0f, 0.5f, 0.0f};
    float camera_distance_initial_{6.0f};
    float camera_yaw_initial_{-120.0f};
    float camera_pitch_initial_{-20.0f};
    float min_camera_distance_ = 1.0f;
    float max_camera_distance_ = 50.0f;
    glm::vec3 world_up_{0.0f, 1.0f, 0.0f};

    bool rotating_ = false;
    bool panning_ = false;
    bool tracking_mouse_ = false;
    double last_x_ = 0.0;
    double last_y_ = 0.0;
    float mouse_rotate_sensitivity_ = 0.2f;
    float mouse_pan_sensitivity_ = 0.002f;
    float scroll_sensitivity_ = 0.8f;

    glm::vec3 ambient_color_{0.2f, 0.2f, 0.2f};
    glm::vec3 light_color_{1.0f, 1.0f, 1.0f};
    glm::vec3 light_direction_{-0.5f, -1.0f, -0.3f};
    float specular_strength_ = 0.6f;
    float shininess_ = 32.0f;
    bool show_axes_ = true;
    float axis_length_ = 2.0f;
    float axis_line_width_ = 2.0f;
    glm::mat4 axis_alignment_{1.0f};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObjViewerNode>();
    node->run();
    rclcpp::shutdown();
    return 0;
}
