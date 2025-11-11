// surround_vision_node.cpp
// 
// Vehicle Coordinate System:
// - X: Forward (positive)
// - Y: Left (positive) 
// - Z: Up (positive)
// - Origin: Center of the rear axle
//
// Virtual Camera:
// - Initial position: [0, 0, 10] looking downward
// - Yaw: Rotation around Z axis
// - Pitch: Rotation around Y axis (looking up/down)
// - Controls: RViz-style mouse controls + keyboard
//
// Vehicle Model Format:
// - Recommended: .obj with .mtl material files (supports textures)
// - Alternative: .fbx (good texture support)
// - Current: .obj (full texture support with tiny_obj_loader)
// - Export from Blender: File -> Export -> Wavefront (.obj) with "Write Materials" checked
// - Scale model to match your vehicle dimensions in Blender before export
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <queue>
#include <memory>
#include <unordered_map>
// #include <ncurses.h>
#include <condition_variable>
#include <fstream>

// OBJ loading and texture support
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Image decoder with libturbojpeg support
#include "image_decoder.h"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "rect_warper.hpp"

namespace surround_vision {

// Structure for OBJ mesh
struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei index_count = 0;
    GLuint texture_id = 0;
};

// Camera configuration structure
struct CameraConfig {
    std::string name;
    // Extrinsic parameters
    glm::mat4 transformation_matrix;  // 4x4 homogeneous transformation
    // Intrinsic parameters
    cv::Mat K;  // Camera matrix
    cv::Mat D;  // Distortion coefficients
    int width, height;
    double fovH;
    // IPM parameters (keeping for now, will remove later)
    // Cached matrices
    cv::Mat undistortMapX, undistortMapY;
    cv::Mat ipmTransform;
    cv::Mat ipmTransformInv;
    // Transformation matrices for 3D projection
    glm::mat4 extrinsic;  // World to Camera transform (View Matrix)
    glm::mat4 intrinsic;  // Camera intrinsic as 4x4 matrix
    glm::vec3 world_pos;  // Camera position in world coordinates
};

// Structure for bowl mesh vertex
struct BowlVertex {
    glm::vec3 position;      // 3D position in world space
    glm::vec2 texCoord;      // UV coordinates
    int cameraId;            // Which camera this vertex maps to (0-3)
    float blendWeight;       // Weight for blending between cameras
};

// Structure for bowl mesh parameters
struct BowlMeshParams {
    float ground_radius = 8.0f;      // Radius of flat ground area (meters)
    float bowl_radius = 15.0f;       // Maximum radius of bowl (meters)
    float bowl_height = 5.0f;        // Height of bowl walls (meters)
    
    // Number of segments around the bowl's circumference.
    // Higher value gives a smoother circular mesh for the surround view projection.
    int radial_segments = 256*8;       

    // Number of concentric rings for the flat ground region at the center of the bowl mesh.
    // More rings provide finer tessellation for the area directly around the vehicle.
    int ground_rings = 32*8;           

    // Number of vertical rings for the curved wall region of the bowl mesh.
    // This controls the smoothness of the transition from ground to wall in the surround view.
    int wall_rings = 48;             
};

// Virtual camera state
struct VirtualCamera {
    // Vehicle coordinate system: X forward, Y left, Z up
    // Initial position: [0, 0, 10] looking downward
    // glm::vec3 position{-2.17f, 28.22f, 27.91f};
    glm::vec3 position{-6.95f, 5.89f, 7.57f};
    float yaw = -33.4f;      // Rotation around Z axis (up)
    float pitch = 148.8f;   // Looking straight down from above
    float roll = 0.0f;     // Roll around X axis

    // Orbital camera parameters for RViz-style controls
    float distance = 14.36f;  // Distance from focal point
    // glm::vec3 focal_point{0.0f, 0.0f, 0.0f};  // Focal point (origin)
    glm::vec3 focal_point{3.48f, -0.99f, 0.0f};

    // Auto-revolution parameters
    bool auto_revolve = false;
    float revolution_speed = 15.0f;  // degrees per second
    float revolution_angle = 0.0f;   // current revolution angle

    float moveSpeed = 0.5f;
    float rotSpeed = 2.0f;
    std::mutex mutex;
    
    glm::mat4 getViewMatrix() {
        std::lock_guard<std::mutex> lock(mutex);
        
        // Calculate camera position based on spherical coordinates
        float x = distance * cos(glm::radians(yaw)) * cos(glm::radians(pitch));
        float y = distance * sin(glm::radians(yaw)) * cos(glm::radians(pitch));
        float z = distance * sin(glm::radians(pitch));
        
        // Set camera position relative to focal point
        position = focal_point + glm::vec3(x, y, z);
        
        // Calculate up vector with roll consideration
        // Start with Z up and apply roll rotation
        glm::vec3 up(0.0f, 0.0f, 1.0f);
        if (roll != 0.0f) {
            // Apply roll rotation around the view direction
            glm::vec3 view_dir = glm::normalize(focal_point - position);
            glm::vec3 right = glm::normalize(glm::cross(view_dir, up));
            // Rotate up vector around view direction using rotation matrix
            float roll_rad = glm::radians(roll);
            float cos_roll = cos(roll_rad);
            float sin_roll = sin(roll_rad);
            up = up * cos_roll + right * sin_roll;
        }
        
        return glm::lookAt(position, focal_point, up);
    }
    
    void move(float dx, float dy, float dz) {
        std::lock_guard<std::mutex> lock(mutex);
        // Move the focal point for panning
        focal_point += glm::vec3(dx, dy, dz) * moveSpeed;
        printPose();
    }
    
    void rotate(float dyaw, float dpitch, float droll = 0.0f) {
        std::lock_guard<std::mutex> lock(mutex);
        yaw += dyaw * rotSpeed;
        pitch += dpitch * rotSpeed;
        roll += droll * rotSpeed;
        // Clamp pitch to prevent camera from going upside down
        // pitch = glm::clamp(pitch, -89.0f, 89.0f);
        printPose();
    }
    
    void zoom(float dz) {
        std::lock_guard<std::mutex> lock(mutex);
        distance += dz * moveSpeed;
        // Clamp distance to reasonable bounds
        distance = glm::clamp(distance, 2.0f, 50.0f);
        printPose();
    }
    
    void updateAutoRevolution(float delta_time) {
        if (!auto_revolve) return;
        
        std::lock_guard<std::mutex> lock(mutex);
        revolution_angle += revolution_speed * delta_time;
        if (revolution_angle >= 360.0f) {
            revolution_angle -= 360.0f;
        }
        
        // Set yaw to revolution angle, keep current pitch and distance
        yaw = revolution_angle;
        // Update position based on spherical coordinates
        float yaw_rad = glm::radians(yaw);
        float pitch_rad = glm::radians(pitch);
        
        position.x = focal_point.x + distance * cos(pitch_rad) * cos(yaw_rad);
        position.y = focal_point.y + distance * cos(pitch_rad) * sin(yaw_rad);
        position.z = focal_point.z + distance * sin(pitch_rad);
    }
    
    void stopAutoRevolution() {
        std::lock_guard<std::mutex> lock(mutex);
        auto_revolve = false;
    }
    
    void startAutoRevolution() {
        std::lock_guard<std::mutex> lock(mutex);
        auto_revolve = true;
        // Keep current position but start revolving from here
        revolution_angle = yaw;
    }

private:
    void printPose() const {
        // This function is called from within a lock
        RCLCPP_INFO(rclcpp::get_logger("virtual_camera"), 
                   "Camera Pose Updated: pos(%.2f, %.2f, %.2f), yaw(%.2f), pitch(%.2f), roll(%.2f), distance(%.2f), focal(%.2f, %.2f, %.2f)",
                   position.x, position.y, position.z, yaw, pitch, roll, distance, 
                   focal_point.x, focal_point.y, focal_point.z);
    }
};

// Mouse control state for RViz-style controls
struct MouseControl {
    bool first_mouse = true;
    double last_x = 0.0;
    double last_y = 0.0;
    bool left_pressed = false;
    bool middle_pressed = false;
    bool right_pressed = false;
    float mouse_sensitivity = 0.1f;
    float zoom_sensitivity = 0.5f;
    std::mutex mutex;
};

class SurroundVisionNode : public rclcpp::Node {
public:
    SurroundVisionNode() : Node("surround_vision_node") {
        RCLCPP_INFO(this->get_logger(), "Initializing Surround Vision System");
        
        // Initialize package share directory
        package_share_directory_ = ament_index_cpp::get_package_share_directory("surround_vision");
        
        // Declare and get parameters
        this->declare_parameter("use_compressed_images", false);
        this->declare_parameter("model_path", "");
        this->declare_parameter("camera_config_path", "");
        this->declare_parameter("windowed_x", 0);
        this->declare_parameter("windowed_y", 400);
        this->declare_parameter("windowed_width", 480);
        this->declare_parameter("windowed_height", 400);
        this->declare_parameter("icon_path", "");
        
        use_compressed_images_ = this->get_parameter("use_compressed_images").as_bool();
        model_path_ = this->get_parameter("model_path").as_string();
        camera_config_path_ = this->get_parameter("camera_config_path").as_string();
        windowed_x_ = this->get_parameter("windowed_x").as_int();
        windowed_y_ = this->get_parameter("windowed_y").as_int();
        windowed_width_ = this->get_parameter("windowed_width").as_int();
        windowed_height_ = this->get_parameter("windowed_height").as_int();
        icon_path_ = this->get_parameter("icon_path").as_string();
        
        // Initialize image decoder with libturbojpeg support
        image_decoder_ = std::make_unique<ImageDecoder>(this->get_logger());
        
        // Provide fallback paths if parameters are not set
        if (model_path_.empty()) {
            model_path_ = package_share_directory_ + "/models/smartCar.obj";
            RCLCPP_WARN(this->get_logger(), "Model path not provided, using default: %s", model_path_.c_str());
        }
        
        if (camera_config_path_.empty()) {
            camera_config_path_ = package_share_directory_ + "/../coenc/config/camerasParam2.yaml";
            RCLCPP_WARN(this->get_logger(), "Camera config path not provided, using default: %s", camera_config_path_.c_str());
        }
        
        if (icon_path_.empty()) {
            icon_path_ = package_share_directory_ + "/icons/surround_vision_icon.png";
            RCLCPP_INFO(this->get_logger(), "Icon path not provided, using default: %s", icon_path_.c_str());
        }
        
        RCLCPP_INFO(this->get_logger(), "Using %s images", 
                   use_compressed_images_ ? "compressed" : "raw decoded");
        RCLCPP_INFO(this->get_logger(), "Front images are considered fisheye distorted");

                   
        RCLCPP_INFO(this->get_logger(), "Model path: %s", model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Camera config path: %s", camera_config_path_.c_str());
        
        // Load camera configurations
        loadCameraConfigurations();
        
        // Initialize subscribers
        initializeSubscribers();
        
        // Start rendering thread
        rendering_thread_ = std::thread(&SurroundVisionNode::renderLoop, this);

        // Wait for OpenGL to be initialized
        std::unique_lock<std::mutex> lock(opengl_mutex_);
        opengl_cv_.wait(lock, [this] { return opengl_initialized_.load(); });
        
        RCLCPP_INFO(this->get_logger(), "Surround Vision System initialized successfully");
        RCLCPP_INFO(this->get_logger(), "Window positioned at (%d, %d) with size %dx%d", 
                   windowed_x_, windowed_y_, windowed_width_, windowed_height_);
        RCLCPP_INFO(this->get_logger(), "Press 'F' key to toggle fullscreen mode");
    }
    
    ~SurroundVisionNode() {
        should_exit_ = true;
        if (rendering_thread_.joinable()) rendering_thread_.join();
        if (input_thread_.joinable()) input_thread_.join();
        
        // Cleanup OpenGL
        if (window_) {
            glfwDestroyWindow(window_);
            glfwTerminate();
        }
    }
    
    // Public method to check if the window should close
    bool shouldExit() const {
        return should_exit_.load();
    }

private:
    // Camera configurations
    std::unordered_map<std::string, CameraConfig> cameras_;
    
    // Image buffers - optimized for production
    std::unordered_map<std::string, cv::Mat> distorted_images_;  // Pre-allocated distorted buffers
    std::unordered_map<std::string, cv::Mat> undistorted_images_;  // Pre-allocated undistorted buffers
    std::unordered_map<std::string, std::atomic<bool>> camera_updated_;  // Per-camera update flags
    // No mutex needed since we process images immediately without storing
    
    // Subscribers
    std::vector<rclcpp::SubscriptionBase::SharedPtr> subscribers_;
    
    // Image type configuration
    bool use_compressed_images_{false};
    
    // Image decoder with libturbojpeg support
    std::unique_ptr<ImageDecoder> image_decoder_;
    
    // OpenGL resources
    GLFWwindow* window_ = nullptr;
    GLuint shader_program_;
    GLuint bowl_shader_program_;  // Specialized shader for bowl mesh
    GLuint vao_, vbo_, ebo_;
    
    // Icon path
    std::string icon_path_;
    
    
    // Window state management
    bool is_fullscreen_ = false;
    int windowed_x_;
    int windowed_y_;
    int windowed_width_;
    int windowed_height_;
    
    // Bowl mesh resources
    GLuint bowl_vao_, bowl_vbo_, bowl_ebo_;
    std::vector<BowlVertex> bowl_vertices_;
    std::vector<unsigned int> bowl_indices_;
    BowlMeshParams bowl_params_;
    
    // Camera textures (4 cameras)
    GLuint camera_textures_[4];  // front, left, right, rear
    bool texture_initialized_[4] = {false, false, false, false};
    
    // Inside camera texture
    GLuint inside_camera_texture_ = 0;
    bool inside_camera_texture_initialized_ = false;
    cv::Mat inside_camera_image_;
    std::atomic<bool> inside_camera_updated_{false};
    
    // Inside camera planar mesh
    GLuint inside_camera_vao_, inside_camera_vbo_, inside_camera_ebo_;
    GLuint inside_camera_shader_program_;
    
    // Ground camera texture
    GLuint ground_camera_texture_ = 0;
    bool ground_camera_texture_initialized_ = false;
    cv::Mat ground_camera_image_;
    std::atomic<bool> ground_camera_updated_{false};
    
    // Ground camera planar mesh
    GLuint ground_camera_vao_, ground_camera_vbo_, ground_camera_ebo_;
    GLuint ground_camera_shader_program_;
    
    // 3D Model (OBJ format)
    std::vector<Mesh> model_meshes_;
    std::string package_share_directory_;
    std::string model_path_;
    std::string camera_config_path_;
    
    // Virtual camera
    VirtualCamera virtual_camera_;
    
    // Mouse control
    MouseControl mouse_control_;
    
    // XYZ axes rendering
    GLuint axes_vao_, axes_vbo_, axes_ebo_;
    GLuint axes_shader_program_;
    
    // Control flags
    std::atomic<bool> should_exit_{false};
    std::atomic<bool> needs_update_{true};
    
    // Threads
    std::thread rendering_thread_;
    std::thread input_thread_;
    
    // BEV dimensions
    const int BEV_WIDTH = 1920;
    const int BEV_HEIGHT = 1080;

    std::atomic<bool> opengl_initialized_{false};
    std::condition_variable opengl_cv_;
    std::mutex opengl_mutex_;
    
    void loadCameraConfigurations() {
        if (camera_config_path_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Camera config path not provided. Please set the 'camera_config_path' parameter.");
            return;
        }
        
        // Check if camera config file exists
        std::ifstream config_file(camera_config_path_);
        if (!config_file.good()) {
            RCLCPP_ERROR(this->get_logger(), "Camera config file does not exist: %s", camera_config_path_.c_str());
            return;
        }
        config_file.close();
        
        try {
            YAML::Node config = YAML::LoadFile(camera_config_path_);
            
            std::vector<std::string> camera_names = {"frontCamera", "leftCamera", "rightCamera", "rearCamera"};
            
            for (const auto& cam_name : camera_names) {
                if (!config[cam_name]) continue;
                
                CameraConfig cam_cfg;
                cam_cfg.name = cam_name;
                
                auto params = config[cam_name]["ros__parameters"];
                               
                // Load 4x4 transformation matrix directly
                auto transform_vals = params["transformation_matrix"].as<std::vector<double>>();
                cam_cfg.transformation_matrix = glm::make_mat4(transform_vals.data());

                // Intrinsic parameters
                auto K_vals = params["intrinsic_matrix"].as<std::vector<double>>();
                cam_cfg.K = cv::Mat(3, 3, CV_64F, K_vals.data()).clone();
                
                auto D_vals = params["distortion_coefficients"].as<std::vector<double>>();
                cam_cfg.D = cv::Mat(1, 4, CV_64F, D_vals.data()).clone();
                
                // Image dimensions
                if (params["image_width"]) {
                    cam_cfg.width = params["image_width"].as<int>();
                    cam_cfg.height = params["image_height"].as<int>();
                } else {
                    cam_cfg.width = 1280;
                    cam_cfg.height = 720;
                }
                
                cam_cfg.fovH = params["field_of_view_horizontal"].as<double>();
                
                computeUndistortionMaps(cam_cfg);
                computeCameraMatrices(cam_cfg);
                
                cameras_[cam_name] = cam_cfg;
                
                RCLCPP_INFO(this->get_logger(), "Loaded configuration for %s", cam_name.c_str());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load camera configurations: %s", e.what());
            throw;
        }
    }
    
    void computeUndistortionMaps(CameraConfig& cam) {

        cv::Size image_size(cam.width, cam.height);
        
        cv::Mat new_K = cam.K.clone();
        double balance = 1.0;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
            cam.K, cam.D, image_size, cv::Mat::eye(3, 3, CV_64F), 
            new_K, balance, image_size);

        cv::fisheye::initUndistortRectifyMap(
            cam.K, cam.D, cv::Mat::eye(3, 3, CV_64F), new_K,
            image_size, CV_32FC1,
            cam.undistortMapX, cam.undistortMapY
        );
        
        cam.K = new_K.clone();
    }
    
    
    void computeCameraMatrices(CameraConfig& cam) {
        // The transformation_matrix in camerasParam2.yaml is vehicle-to-camera
        // We need to convert it to world-to-camera (extrinsic matrix)
        
        // Extract camera position from transformation matrix (4th column)
        cam.world_pos = glm::vec3(cam.transformation_matrix[0][3], cam.transformation_matrix[1][3], cam.transformation_matrix[2][3]);
        
        // Create lookAt matrix for world-to-camera transformation
        // The camera looks in the direction of its Z-axis (forward in camera coordinates)
        glm::vec3 cam_forward = glm::normalize(glm::vec3(cam.transformation_matrix[0][2], cam.transformation_matrix[1][2], cam.transformation_matrix[2][2]));
        glm::vec3 cam_up = glm::normalize(glm::vec3(cam.transformation_matrix[0][1], cam.transformation_matrix[1][1], cam.transformation_matrix[2][1]));
        cam_up *= -1;   // Otherwise, all the camera images are looking down rotated 180 degrees
        glm::vec3 look_at_point = cam.world_pos + cam_forward;
        
        // Create the extrinsic matrix (world-to-camera)
        cam.extrinsic = glm::lookAt(cam.world_pos, look_at_point, cam_up);
        
        // Print camera information for debugging
        RCLCPP_INFO(this->get_logger(), "Camera %s: pos(%.3f, %.3f, %.3f), forward(%.3f, %.3f, %.3f), up(%.3f, %.3f, %.3f)", 
                   cam.name.c_str(), cam.world_pos.x, cam.world_pos.y, cam.world_pos.z,
                   cam_forward.x, cam_forward.y, cam_forward.z,
                   cam_up.x, cam_up.y, cam_up.z);
    }

    void initializeSubscribers() {
        auto qos = rclcpp::QoS(10).best_effort();
        
        if (use_compressed_images_) {
            // Subscribe to compressed images
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::CompressedImage>(
                "/frontCamera/v4l2/compressed", qos,
                [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    processCompressedImage("frontCamera", msg);
                }));
            
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::CompressedImage>(
                "/leftCamera/v4l2/compressed", qos,
                [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    processCompressedImage("leftCamera", msg);
                }));
            
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::CompressedImage>(
                "/rightCamera/v4l2/compressed", qos,
                [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    processCompressedImage("rightCamera", msg);
                }));
            
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::CompressedImage>(
                "/rearCamera/v4l2/compressed", qos,
                [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    processCompressedImage("rearCamera", msg);
                }));
        } else {
            // Subscribe to raw decoded images
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::Image>(
                "/frontCamera/decoded", qos,
                [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                    processRawImage("frontCamera", msg);
                }));
            
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::Image>(
                "/leftCamera/decoded", qos,
                [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                    processRawImage("leftCamera", msg);
                }));
            
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::Image>(
                "/rightCamera/decoded", qos,
                [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                    processRawImage("rightCamera", msg);
                }));
            
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::Image>(
                "/rearCamera/decoded", qos,
                [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                    processRawImage("rearCamera", msg);
                }));
        }
        
        // Add inside camera subscriber
        if (use_compressed_images_) {
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::CompressedImage>(
                "/insideCamera/v4l2/compressed", qos,
                [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    processInsideCameraCompressed(msg);
                }));
        } else {
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::Image>(
                "/insideCamera/decoded", qos,
                [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                    processInsideCameraRaw(msg);
                }));
        }
        
        // Add ground camera subscriber
        if (use_compressed_images_) {
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::CompressedImage>(
                "/groundCamera/v4l2/compressed", qos,
                [this](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                    processGroundCameraCompressed(msg);
                }));
        } else {
            subscribers_.push_back(this->create_subscription<sensor_msgs::msg::Image>(
                "/groundCamera/decoded", qos,
                [this](const sensor_msgs::msg::Image::SharedPtr msg) {
                    processGroundCameraRaw(msg);
                }));
        }
    }
    
    void processCompressedImage(const std::string& camera_name, const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        try {
            // Use the efficient ImageDecoder with libturbojpeg support
            // cv::Mat image = image_decoder_->decode(msg);
            cv::Mat& image = distorted_images_[camera_name];
            image = image_decoder_->decode(msg);
            if (image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Failed to decode image from %s", camera_name.c_str());
                return;
            }
            
            // Optimized: Direct processing without unnecessary copying
            processImageOptimized(camera_name, image);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing image from %s: %s", 
                        camera_name.c_str(), e.what());
        }
    }
    
    void processRawImage(const std::string& camera_name, const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // Optimized: Use toCvShare to avoid copying when possible
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
            if (cv_ptr->image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Failed to convert image from %s", camera_name.c_str());
                return;
            }
            
            // Optimized: Direct processing without unnecessary copying
            processImageOptimized(camera_name, cv_ptr->image);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing raw image from %s: %s", 
                        camera_name.c_str(), e.what());
        }
    }
    
    // Optimized image processing function - performs undistortion immediately
    void processImageOptimized(const std::string& camera_name, const cv::Mat& image) {
        // Perform undistortion immediately to avoid storing distorted images
        auto& cam_cfg = cameras_[camera_name];
        cv::Mat& undistorted = undistorted_images_[camera_name];  // Use pre-allocated buffer
        
        // Perform undistortion in-place to avoid memory allocation
        cv::remap(image, undistorted, cam_cfg.undistortMapX, cam_cfg.undistortMapY, 
                 cv::INTER_LINEAR);
        
        // Set per-camera update flag (lock-free)
        camera_updated_[camera_name] = true;
        needs_update_ = true;
    }
    
    void processInsideCameraCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        try {
            // Use the efficient ImageDecoder with libturbojpeg support
            inside_camera_image_ = image_decoder_->decode(msg);
            if (inside_camera_image_.empty()) {
                RCLCPP_WARN(this->get_logger(), "Failed to decode inside camera image");
                return;
            }
            
            // Set update flag (lock-free)
            inside_camera_updated_ = true;
            needs_update_ = true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing inside camera compressed image: %s", e.what());
        }
    }
    
    void processInsideCameraRaw(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // Use toCvShare to avoid copying when possible
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
            if (cv_ptr->image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Failed to convert inside camera image");
                return;
            }
            
            // Copy the image
            inside_camera_image_ = cv_ptr->image.clone();
            
            // Set update flag (lock-free)
            inside_camera_updated_ = true;
            needs_update_ = true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing inside camera raw image: %s", e.what());
        }
    }

    void updateGroundCameraImage(const cv::Mat& image, const char* empty_warning, const char* size_warning) {
        if (image.empty()) {
            RCLCPP_WARN(this->get_logger(), "%s", empty_warning);
            return;
        }

        const cv::Size target_size = ground_camera_image_.size();
        const bool has_target = target_size.height > 0 && target_size.width > 0;

        if (has_target && image.rows >= target_size.height && image.cols >= target_size.width) {
            cv::Rect roi(0, image.rows - target_size.height, target_size.width, target_size.height);
            ground_camera_image_ = image(roi).clone();
        } else {
            ground_camera_image_ = image.clone();
            if (has_target) {
                RCLCPP_WARN(this->get_logger(), "%s", size_warning);
            }
        }

        // Set update flag (lock-free)
        ground_camera_updated_ = true;
        needs_update_ = true;
    }
    
    void processGroundCameraCompressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        try {
            // Use the efficient ImageDecoder with libturbojpeg support 
            cv::Mat groundCamImg = image_decoder_->decode(msg);
            updateGroundCameraImage( groundCamImg, "Failed to decode ground camera image", "Decoded ground camera image too small, skipping crop");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing ground camera compressed image: %s", e.what());
        }
    }
    
    void processGroundCameraRaw(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // Use toCvShare to avoid copying when possible
            cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
            updateGroundCameraImage( cv_ptr->image, "Failed to convert ground camera image", "Raw ground camera image too small, skipping crop");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing ground camera raw image: %s", e.what());
        }
    }
    
    void initializeOpenGL() {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        
        // Create window with initial position and size for teleoperation
        window_ = glfwCreateWindow(windowed_width_, windowed_height_, "Surround Vision", NULL, NULL);
        if (!window_) {
            glfwTerminate();
            throw std::runtime_error("Failed to create GLFW window");
        }
        
        // Set window title for desktop environment identification
        glfwSetWindowTitle(window_, "Surround Vision");
        
        // Set window position for teleoperation layout
        glfwSetWindowPos(window_, windowed_x_, windowed_y_);
        
        // Load and set window icon
        loadWindowIcon();
        
        glfwMakeContextCurrent(window_);
        glfwSetWindowUserPointer(window_, this);
        glfwSetKeyCallback(window_, keyCallback);
        glfwSetMouseButtonCallback(window_, mouseButtonCallback);
        glfwSetCursorPosCallback(window_, mouseCursorCallback);
        glfwSetScrollCallback(window_, scrollCallback);
        glfwSetWindowCloseCallback(window_, windowCloseCallback);
        glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
        
        if (glewInit() != GLEW_OK) {
            throw std::runtime_error("Failed to initialize GLEW");
        }
        
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        createShaders();
        createAxesShaders();
        createInsideCameraShaders();
        createGroundCameraShaders();
        createQuad();
        createAxes();
        createInsideCameraPlanarMesh();
        createGroundCameraPlanarMesh();
        initializeCameraTextures();
        initializeInsideCameraTexture();
        initializeGroundCameraTexture();
        generateBowlMesh();
    }
    
    void createShaders() {
        createStandardShaders();
        createBowlShaders();
    }
    
    void createStandardShaders() {
        const char* vertex_shader_src = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec3 aNormal;
            layout (location = 2) in vec2 aTexCoord;
            
            out vec2 TexCoord;
            out vec3 Normal;
            out vec3 FragPos;
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            void main() {
                FragPos = vec3(model * vec4(aPos, 1.0));
                Normal = mat3(transpose(inverse(model))) * aNormal;
                TexCoord = aTexCoord;
                
                gl_Position = projection * view * model * vec4(aPos, 1.0);
            }
        )";
        
        const char* fragment_shader_src = R"(
            #version 330 core
            out vec4 FragColor;
            
            in vec2 TexCoord;
            in vec3 Normal;
            in vec3 FragPos;
            
            uniform sampler2D texture1;
            uniform float alpha;
            
            // Lighting uniforms
            uniform vec3 lightPos;
            uniform vec3 viewPos;
            uniform vec3 lightColor;
            uniform vec3 ambientColor;
            
            void main() {
                vec4 texColor = texture(texture1, TexCoord);
                if (texColor.a < 0.1) {
                    discard;
                }
                
                // Ambient lighting
                vec3 ambient = ambientColor * texColor.rgb;
                
                // Diffuse lighting
                vec3 norm = normalize(Normal);
                vec3 lightDir = normalize(lightPos - FragPos);
                float diff = max(dot(norm, lightDir), 0.0);
                vec3 diffuse = diff * lightColor * texColor.rgb;
                
                // Specular lighting
                vec3 viewDir = normalize(viewPos - FragPos);
                vec3 reflectDir = reflect(-lightDir, norm);
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
                vec3 specular = spec * lightColor * texColor.rgb;
                
                vec3 result = ambient + diffuse + specular;
                FragColor = vec4(result, alpha);
            }
        )";
        
        GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_src, NULL);
        glCompileShader(vertex_shader);
        
        GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_src, NULL);
        glCompileShader(fragment_shader);
        
        shader_program_ = glCreateProgram();
        glAttachShader(shader_program_, vertex_shader);
        glAttachShader(shader_program_, fragment_shader);
        glLinkProgram(shader_program_);
        
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
    }
    
    void createBowlShaders() {
        const char* bowl_vertex_shader_src = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec2 aTexCoord;
            layout (location = 2) in int aCameraId;
            layout (location = 3) in float aBlendWeight;
            
            out vec2 TexCoord;
            flat out int CameraId;
            out float BlendWeight;
            out vec3 WorldPos;
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            void main() {
                vec4 worldPos = model * vec4(aPos, 1.0);
                WorldPos = worldPos.xyz;
                gl_Position = projection * view * worldPos;
                TexCoord = aTexCoord;
                CameraId = aCameraId;
                BlendWeight = aBlendWeight;
            }
        )";
        
        const char* bowl_fragment_shader_src = R"(
            #version 330 core
            out vec4 FragColor;
            
            in vec2 TexCoord;
            flat in int CameraId;
            in float BlendWeight;
            in vec3 WorldPos;
            
            uniform sampler2D cameraTextures[4];
            uniform bool textureValid[4];
            uniform float alpha;
            
            vec3 sampleCameraTexture(int camId, vec2 uv) {
                if (camId < 0 || camId >= 4) {
                    return vec3(0.1, 0.1, 0.1);
                }
                
                if (!textureValid[camId]) {
                    float gradient = uv.y;
                    if (camId == 0) { return vec3(0.9 - gradient * 0.4, 0.1 + gradient * 0.2, 0.1); } // Front - red
                    if (camId == 1) { return vec3(0.1, 0.9 - gradient * 0.4, 0.1 + gradient * 0.2); } // Left - green
                    if (camId == 2) { return vec3(0.1 + gradient * 0.2, 0.1, 0.9 - gradient * 0.4); } // Right - blue
                    return vec3(0.9 - gradient * 0.3, 0.9 - gradient * 0.3, 0.1 + gradient * 0.3); // Rear - yellow
                }
                
                vec2 final_uv = uv;
                
                // Apply horizontal flip for rear camera (camId == 3) if enabled
                if (camId == 3) {
                    final_uv.x = 1.0 - uv.x;
                }
                
                vec3 color;
                if (camId == 0) color = texture(cameraTextures[0], final_uv).rgb;
                else if (camId == 1) color = texture(cameraTextures[1], final_uv).rgb;
                else if (camId == 2) color = texture(cameraTextures[2], final_uv).rgb;
                else color = texture(cameraTextures[3], final_uv).rgb;
                
                return color;
            }
            
            void main() {
                vec3 color = sampleCameraTexture(CameraId, TexCoord);
                FragColor = vec4(color, alpha);
            }
        )";
        
        GLuint bowl_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(bowl_vertex_shader, 1, &bowl_vertex_shader_src, NULL);
        glCompileShader(bowl_vertex_shader);
        
        GLint success;
        GLchar infoLog[512];
        glGetShaderiv(bowl_vertex_shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(bowl_vertex_shader, 512, NULL, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Bowl vertex shader compilation failed: %s", infoLog);
        }
        
        GLuint bowl_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(bowl_fragment_shader, 1, &bowl_fragment_shader_src, NULL);
        glCompileShader(bowl_fragment_shader);
        
        glGetShaderiv(bowl_fragment_shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(bowl_fragment_shader, 512, NULL, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Bowl fragment shader compilation failed: %s", infoLog);
        }
        
        bowl_shader_program_ = glCreateProgram();
        glAttachShader(bowl_shader_program_, bowl_vertex_shader);
        glAttachShader(bowl_shader_program_, bowl_fragment_shader);
        glLinkProgram(bowl_shader_program_);
        
        glGetProgramiv(bowl_shader_program_, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(bowl_shader_program_, 512, NULL, infoLog);
            RCLCPP_ERROR(this->get_logger(), "Bowl shader program linking failed: %s", infoLog);
        }
        
        glDeleteShader(bowl_vertex_shader);
        glDeleteShader(bowl_fragment_shader);
    }
    
    void createAxesShaders() {
        const char* axes_vertex_shader_src = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec3 aColor;
            
            out vec3 Color;
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            void main() {
                gl_Position = projection * view * model * vec4(aPos, 1.0);
                Color = aColor;
            }
        )";
        
        const char* axes_fragment_shader_src = R"(
            #version 330 core
            out vec4 FragColor;
            
            in vec3 Color;
            
            void main() {
                FragColor = vec4(Color, 1.0);
            }
        )";
        
        GLuint axes_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(axes_vertex_shader, 1, &axes_vertex_shader_src, NULL);
        glCompileShader(axes_vertex_shader);
        
        GLuint axes_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(axes_fragment_shader, 1, &axes_fragment_shader_src, NULL);
        glCompileShader(axes_fragment_shader);
        
        axes_shader_program_ = glCreateProgram();
        glAttachShader(axes_shader_program_, axes_vertex_shader);
        glAttachShader(axes_shader_program_, axes_fragment_shader);
        glLinkProgram(axes_shader_program_);
        
        glDeleteShader(axes_vertex_shader);
        glDeleteShader(axes_fragment_shader);
    }
    
    void createInsideCameraShaders() {
        const char* inside_camera_vertex_shader_src = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec2 aTexCoord;
            
            out vec2 TexCoord;
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            void main() {
                gl_Position = projection * view * model * vec4(aPos, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        
        const char* inside_camera_fragment_shader_src = R"(
            #version 330 core
            out vec4 FragColor;
            
            in vec2 TexCoord;
            
            uniform sampler2D insideCameraTexture;
            uniform bool textureValid;
            uniform float alpha;
            
            void main() {
                if (!textureValid) {
                    // Show a placeholder color when texture is not available
                    FragColor = vec4(0.2f, 0.2f, 0.2f, alpha);
                    return;
                }
                
                vec4 texColor = texture(insideCameraTexture, TexCoord);
                FragColor = vec4(texColor.rgb, alpha);
            }
        )";
        
        GLuint inside_camera_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(inside_camera_vertex_shader, 1, &inside_camera_vertex_shader_src, NULL);
        glCompileShader(inside_camera_vertex_shader);
        
        GLuint inside_camera_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(inside_camera_fragment_shader, 1, &inside_camera_fragment_shader_src, NULL);
        glCompileShader(inside_camera_fragment_shader);
        
        inside_camera_shader_program_ = glCreateProgram();
        glAttachShader(inside_camera_shader_program_, inside_camera_vertex_shader);
        glAttachShader(inside_camera_shader_program_, inside_camera_fragment_shader);
        glLinkProgram(inside_camera_shader_program_);
        
        glDeleteShader(inside_camera_vertex_shader);
        glDeleteShader(inside_camera_fragment_shader);
        
        RCLCPP_INFO(this->get_logger(), "Inside camera shaders created successfully");
    }
    
    void createGroundCameraShaders() {
        const char* ground_camera_vertex_shader_src = R"(
            #version 330 core
            layout (location = 0) in vec3 aPos;
            layout (location = 1) in vec2 aTexCoord;
            
            out vec2 TexCoord;
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            void main() {
                gl_Position = projection * view * model * vec4(aPos, 1.0);
                TexCoord = aTexCoord;
            }
        )";
        
        const char* ground_camera_fragment_shader_src = R"(
            #version 330 core
            out vec4 FragColor;
            
            in vec2 TexCoord;
            
            uniform sampler2D groundCameraTexture;
            uniform bool textureValid;
            uniform float alpha;
            
            void main() {
                if (!textureValid) {
                    // Show a placeholder color when texture is not available
                    FragColor = vec4(0.1f, 0.1f, 0.1f, alpha);
                    return;
                }
                
                vec4 texColor = texture(groundCameraTexture, TexCoord);
                FragColor = vec4(texColor.rgb, alpha);
            }
        )";
        
        GLuint ground_camera_vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(ground_camera_vertex_shader, 1, &ground_camera_vertex_shader_src, NULL);
        glCompileShader(ground_camera_vertex_shader);
        
        GLuint ground_camera_fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(ground_camera_fragment_shader, 1, &ground_camera_fragment_shader_src, NULL);
        glCompileShader(ground_camera_fragment_shader);
        
        ground_camera_shader_program_ = glCreateProgram();
        glAttachShader(ground_camera_shader_program_, ground_camera_vertex_shader);
        glAttachShader(ground_camera_shader_program_, ground_camera_fragment_shader);
        glLinkProgram(ground_camera_shader_program_);
        
        glDeleteShader(ground_camera_vertex_shader);
        glDeleteShader(ground_camera_fragment_shader);
        
        RCLCPP_INFO(this->get_logger(), "Ground camera shaders created successfully");
    }
    
    void createQuad() {
        float vertices[] = {
            -1.0f,  1.0f, 0.0f,  0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f,  0.0f, 0.0f,
             1.0f, -1.0f, 0.0f,  1.0f, 0.0f,
             1.0f,  1.0f, 0.0f,  1.0f, 1.0f
        };
        
        unsigned int indices[] = { 0, 1, 2, 0, 2, 3 };
        
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_);
        glGenBuffers(1, &ebo_);
        
        glBindVertexArray(vao_);
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);
    }
    
    void generateBowlMesh() {
        bowl_vertices_.clear();
        bowl_indices_.clear();
        
        const float ground_radius = bowl_params_.ground_radius;
        const float bowl_radius = bowl_params_.bowl_radius;
        const float bowl_height = bowl_params_.bowl_height;
        const int radial_segments = bowl_params_.radial_segments;
        const int ground_rings = bowl_params_.ground_rings;
        const int wall_rings = bowl_params_.wall_rings;
        
        bowl_vertices_.push_back({glm::vec3(0, 0, 0), glm::vec2(0.5f, 0.5f), -1, 1.0f});
        
        for (int ring = 1; ring <= ground_rings; ring++) {
            float r = (float)ring / ground_rings * ground_radius;
            for (int seg = 0; seg < radial_segments; seg++) {
                float theta = 2.0f * M_PI * seg / radial_segments;
                float x = r * cos(theta);
                float y = r * sin(theta);
                bowl_vertices_.push_back({glm::vec3(x, y, 0), glm::vec2(0, 0), -1, 1.0f});
            }
        }
        
        for (int ring = 1; ring <= wall_rings; ring++) {
            float t = (float)ring / wall_rings;
            float bowl_t = t * t;
            float r = ground_radius + (bowl_radius - ground_radius) * t;
            float z = bowl_height * bowl_t;
            
            for (int seg = 0; seg < radial_segments; seg++) {
                float theta = 2.0f * M_PI * seg / radial_segments;
                float x = r * cos(theta);
                float y = r * sin(theta);
                bowl_vertices_.push_back({glm::vec3(x, y, z), glm::vec2(0, 0), -1, 1.0f});
            }
        }
        
        for (int seg = 0; seg < radial_segments; seg++) {
            int next_seg = (seg + 1) % radial_segments;
            bowl_indices_.push_back(0);
            bowl_indices_.push_back(1 + seg);
            bowl_indices_.push_back(1 + next_seg);
        }
        
        int total_rings = ground_rings + wall_rings;
        for (int ring = 0; ring < total_rings - 1; ring++) {
            for (int seg = 0; seg < radial_segments; seg++) {
                int next_seg = (seg + 1) % radial_segments;
                int curr_ring_base = 1 + ring * radial_segments;
                int next_ring_base = 1 + (ring + 1) * radial_segments;
                
                bowl_indices_.push_back(curr_ring_base + seg);
                bowl_indices_.push_back(next_ring_base + seg);
                bowl_indices_.push_back(next_ring_base + next_seg);
                
                bowl_indices_.push_back(curr_ring_base + seg);
                bowl_indices_.push_back(next_ring_base + next_seg);
                bowl_indices_.push_back(curr_ring_base + next_seg);
            }
        }
        
        calculateBowlUVMapping();
        createBowlBuffers();
        
        RCLCPP_INFO(this->get_logger(), "Bowl mesh generated with %zu vertices and %zu indices",
                   bowl_vertices_.size(), bowl_indices_.size());
    }
    
    void calculateBowlUVMapping() {
        std::string cam_names[] = {"frontCamera", "leftCamera", "rightCamera", "rearCamera"};
        
        for (auto& vertex : bowl_vertices_) {
            int best_camera = -1;
            float best_score = -999.0f;
            glm::vec2 best_uv(0.5f, 0.5f);
            
            for (int cam_id = 0; cam_id < 4; cam_id++) {
                auto& cam = cameras_[cam_names[cam_id]];
                
                glm::vec4 world_point(vertex.position, 1.0f);
                glm::vec4 cam_point = cam.extrinsic * world_point;
                
                // In the camera's view space (using glm::lookAt), objects in front have a negative Z value.
                // We must filter out any vertices that are behind the camera's near plane.
                if (cam_point.z > -0.01f) {
                    continue;
                }
                
                // Perform perspective division. The distance to the camera is -cam_point.z.
                float x_ndc = cam_point.x / -cam_point.z;
                
                // The camera's Y-axis points up, but OpenCV's image coordinates have Y pointing down.
                // We must negate the Y coordinate before applying the intrinsic matrix.
                float y_ndc = cam_point.y / -cam_point.z;
                
                float u_pixel = cam.K.at<double>(0, 0) * x_ndc + cam.K.at<double>(0, 2);
                float v_pixel = cam.K.at<double>(1, 1) * (-y_ndc) + cam.K.at<double>(1, 2);
                
                float margin = 75;
                if (u_pixel < -margin || u_pixel >= cam.width + margin || 
                    v_pixel < -margin || v_pixel >= cam.height + margin) {
                    continue;
                }
                
                float u_clamped = glm::clamp(u_pixel, 0.0f, (float)cam.width - 1);
                float v_clamped = glm::clamp(v_pixel, 0.0f, (float)cam.height - 1);
                
                // --- Final Corrected Scoring Logic ---
                // 1. Direction score (main factor)
                // The forward vector in a GLM lookAt matrix is the negative Z-axis of the camera's local space.
                // The columns of the inverse view matrix (cam_to_world) are the axes of the camera in world space.
                // So, column 2 (index 2) is the camera's Z-axis (which points behind it). We negate it to get the forward vector.
                glm::mat4 cam_to_world = glm::inverse(cam.extrinsic);
                glm::vec3 cam_forward_world = -glm::normalize(glm::vec3(cam_to_world[2]));

                glm::vec3 cam_to_vertex = glm::normalize(vertex.position - cam.world_pos);
                float direction_score = glm::dot(cam_forward_world, cam_to_vertex);

                if (direction_score < 0.1) {
                    continue; // Must be generally in front of camera
                }

                // 2. Centeredness score
                float center_u = cam.width / 2.0f;
                float center_v = cam.height / 2.0f;
                float dist_from_center = sqrt(pow(u_clamped - center_u, 2) + pow(v_clamped - center_v, 2));
                float max_dist = sqrt(pow(center_u, 2) + pow(center_v, 2));
                float center_score = 1.0f - (dist_from_center / max_dist);

                // Combined score
                float score = direction_score * 0.7f + center_score * 0.3f;
                
                                 if (score > best_score) {
                     best_score = score;
                     best_camera = cam_id;
                     // No flip needed since we eliminated cv::flip() operation
                     best_uv = glm::vec2(u_clamped / cam.width, v_clamped / cam.height);
                 }
            }
            
            vertex.cameraId = best_camera;
            vertex.texCoord = best_uv;
        }
    }
    
    void createBowlBuffers() {
        glGenVertexArrays(1, &bowl_vao_);
        glGenBuffers(1, &bowl_vbo_);
        glGenBuffers(1, &bowl_ebo_);
        
        glBindVertexArray(bowl_vao_);
        
        glBindBuffer(GL_ARRAY_BUFFER, bowl_vbo_);
        glBufferData(GL_ARRAY_BUFFER, bowl_vertices_.size() * sizeof(BowlVertex),
                    bowl_vertices_.data(), GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bowl_ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, bowl_indices_.size() * sizeof(unsigned int),
                    bowl_indices_.data(), GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(BowlVertex), (void*)offsetof(BowlVertex, position));
        glEnableVertexAttribArray(0);
        
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(BowlVertex), (void*)offsetof(BowlVertex, texCoord));
        glEnableVertexAttribArray(1);
        
        glVertexAttribIPointer(2, 1, GL_INT, sizeof(BowlVertex), (void*)offsetof(BowlVertex, cameraId));
        glEnableVertexAttribArray(2);
        
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(BowlVertex), (void*)offsetof(BowlVertex, blendWeight));
        glEnableVertexAttribArray(3);
        
        glBindVertexArray(0);
    }
    
    void createAxes() {
        const float axis_length = 2.0f;
        float vertices[] = {
            0.0f, 0.0f, 0.0f,   1.0f, 0.0f, 0.0f,
            axis_length, 0.0f, 0.0f,   1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,   0.0f, 1.0f, 0.0f,
            0.0f, axis_length, 0.0f,   0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f,   0.0f, 0.0f, 0.8f,
            0.0f, 0.0f, axis_length,   0.0f, 0.0f, 0.8f
        };
        
        unsigned int indices[] = { 0, 1, 2, 3, 4, 5 };
        
        glGenVertexArrays(1, &axes_vao_);
        glGenBuffers(1, &axes_vbo_);
        glGenBuffers(1, &axes_ebo_);
        
        glBindVertexArray(axes_vao_);
        
        glBindBuffer(GL_ARRAY_BUFFER, axes_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);
    }
    
    void createInsideCameraPlanarMesh() {
        // Create a planar mesh for inside camera texture
        // Size: 1.6m x 1.2m (maintaining 640:480 aspect ratio)
        // Position: Z=1.55m (height), centered at X=+0.5m
        // The plane is parallel to XY plane, rotated counter-clockwise by 90 degrees
        
        float width = 1.0f;   // X dimension
        float height = width*3/4;  // Y dimension
        float z_pos = 1.53f;  // Z position
        float x_center = 0.3f; // X center offset
        
        // Counter-clockwise 90-degree rotation: (x,y) -> (y,-x)
        // UV coordinates flipped horizontally to correct image orientation
        float vertices[] = {
            // Position (x, y, z) + Texture coordinates (u, v)
            // After 90deg CCW rotation: original bottom-left becomes top-left
            x_center - height/2, -width/2, z_pos,  1.0f, 1.0f,  // Bottom-left (rotated) - U flipped
            x_center - height/2,  width/2, z_pos,  0.0f, 1.0f,  // Bottom-right (rotated) - U flipped
            x_center + height/2,  width/2, z_pos,  0.0f, 0.0f,  // Top-right (rotated) - U flipped
            x_center + height/2, -width/2, z_pos,  1.0f, 0.0f   // Top-left (rotated) - U flipped
        };
        
        unsigned int indices[] = {
            0, 1, 2,  // First triangle
            0, 2, 3   // Second triangle
        };
        
        glGenVertexArrays(1, &inside_camera_vao_);
        glGenBuffers(1, &inside_camera_vbo_);
        glGenBuffers(1, &inside_camera_ebo_);
        
        glBindVertexArray(inside_camera_vao_);
        
        glBindBuffer(GL_ARRAY_BUFFER, inside_camera_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, inside_camera_ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);
        
        RCLCPP_INFO(this->get_logger(), "Inside camera planar mesh created (1.2x0.9m at Z=1.53m, X=+0.3m)");
    }
    
    void createGroundCameraPlanarMesh() {
        // Create a planar mesh for ground camera texture
        // Size: 1.0m x 0.375m (maintaining 640:240 aspect ratio)
        // Position: X=-0.5m (behind vehicle), bottom edge at Z=0
        // The plane is parallel to YZ plane
        
        float width = 1.35f;    // Y dimension (width of the plane)
        float height = width*3/8; // Z dimension (height of the plane)
        float x_pos = -0.36f;   // X position (behind vehicle)
        float z_bottom = 0.0f; // Bottom edge at ground level
        
        // UV coordinates flipped horizontally to correct image orientation
        float vertices[] = {
            // Position (x, y, z) + Texture coordinates (u, v)
            x_pos, -width/2, z_bottom,         1.0f, 1.0f,  // Bottom-left - U flipped
            x_pos,  width/2, z_bottom,         0.0f, 1.0f,  // Bottom-right - U flipped
            x_pos,  width/2, z_bottom + height, 0.0f, 0.0f,  // Top-right - U flipped
            x_pos, -width/2, z_bottom + height, 1.0f, 0.0f   // Top-left - U flipped
        };
        
        unsigned int indices[] = {
            0, 1, 2,  // First triangle
            0, 2, 3   // Second triangle
        };
        
        glGenVertexArrays(1, &ground_camera_vao_);
        glGenBuffers(1, &ground_camera_vbo_);
        glGenBuffers(1, &ground_camera_ebo_);
        
        glBindVertexArray(ground_camera_vao_);
        
        glBindBuffer(GL_ARRAY_BUFFER, ground_camera_vbo_);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ground_camera_ebo_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
        
        // Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        
        // Texture coordinate attribute
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
        
        glBindVertexArray(0);
        
        RCLCPP_INFO(this->get_logger(), "Ground camera planar mesh created (1.0x0.375m at X=-0.5m, bottom at Z=0)");
    }
    
    GLuint load_texture(const std::string& path) {
        GLuint textureID;
        glGenTextures(1, &textureID);

        int width, height, nrComponents;
        unsigned char *data = stbi_load(path.c_str(), &width, &height, &nrComponents, 0);
        if (data) {
            GLenum format;
            if (nrComponents == 1) format = GL_RED;
            else if (nrComponents == 3) format = GL_RGB;
            else if (nrComponents == 4) format = GL_RGBA;
            else {
                 RCLCPP_ERROR(this->get_logger(), "Unsupported texture format with %d components", nrComponents);
                 stbi_image_free(data);
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
        }
        stbi_image_free(data);

        return textureID;
    }

    void load3DModel() {
        if (model_path_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Model path not provided. Please set the 'model_path' parameter.");
            return;
        }
        
        // Extract directory from model path for MTL files
        std::string mtl_dir = model_path_.substr(0, model_path_.find_last_of("/\\") + 1);
        RCLCPP_INFO(this->get_logger(), "Loading 3D model from: %s", model_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "Looking for MTL files in: %s", mtl_dir.c_str());
        
        // Check if model file exists
        std::ifstream model_file(model_path_);
        if (!model_file.good()) {
            RCLCPP_ERROR(this->get_logger(), "Model file does not exist: %s", model_path_.c_str());
            return;
        }
        model_file.close();
        
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, model_path_.c_str(), mtl_dir.c_str())) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s%s", warn.c_str(), err.c_str());
            return;
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
            if (material_id != -1 && material_id < texture_ids.size()) {
                 mesh.texture_id = texture_ids[material_id];
            } else {
                 mesh.texture_id = 0; // Default or no texture
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
            model_meshes_.push_back(mesh);
        }
        
        RCLCPP_INFO(this->get_logger(), "3D model loaded successfully with %zu meshes", model_meshes_.size());
    }
    
    void loadWindowIcon() {
        if (icon_path_.empty()) {
            RCLCPP_WARN(this->get_logger(), "Icon path not provided, using default window icon");
            return;
        }
        
        // Load PNG image using stb_image
        int width, height, channels;
        unsigned char* data = stbi_load(icon_path_.c_str(), &width, &height, &channels, 4); // Force RGBA
        
        if (!data) {
            RCLCPP_WARN(this->get_logger(), "Failed to load icon from: %s", icon_path_.c_str());
            return;
        }
        
        // Create GLFW image structure
        GLFWimage icon;
        icon.width = width;
        icon.height = height;
        icon.pixels = data;
        
        // Set the window icon
        glfwSetWindowIcon(window_, 1, &icon);
        
        // Free the image data
        stbi_image_free(data);
        
        RCLCPP_INFO(this->get_logger(), "Window icon loaded successfully from: %s", icon_path_.c_str());
    }
    
    void initializeCameraTextures() {
        glGenTextures(4, camera_textures_);
        
        std::string cam_names[] = {"frontCamera", "leftCamera", "rightCamera", "rearCamera"};
        
        for (int i = 0; i < 4; i++) {
            const std::string& cam_name = cam_names[i];
            
            // Initialize OpenGL texture
            glBindTexture(GL_TEXTURE_2D, camera_textures_[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1280, 720, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
            texture_initialized_[i] = false;
            
            // Pre-allocate undistorted image buffer for this camera
            auto& cam_cfg = cameras_[cam_name];
            if (use_compressed_images_)
                distorted_images_[cam_name] = cv::Mat::zeros(cam_cfg.height, cam_cfg.width, CV_8UC3);
            undistorted_images_[cam_name] = cv::Mat::zeros(cam_cfg.height, cam_cfg.width, CV_8UC3);
            
            // Initialize per-camera update flag
            camera_updated_[cam_name] = false;
        }
    }
    
    void initializeInsideCameraTexture() {
        glGenTextures(1, &inside_camera_texture_);
        
        // Initialize OpenGL texture for inside camera (640x480)
        glBindTexture(GL_TEXTURE_2D, inside_camera_texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        inside_camera_texture_initialized_ = false;
        
        // Pre-allocate inside camera image buffer
        inside_camera_image_ = cv::Mat::zeros(480, 640, CV_8UC3);
        
        // Initialize update flag
        inside_camera_updated_ = false;
        
        RCLCPP_INFO(this->get_logger(), "Inside camera texture initialized (640x480)");
    }
    
    void initializeGroundCameraTexture() {
        glGenTextures(1, &ground_camera_texture_);
        
        // Initialize OpenGL texture for ground camera (640x240)
        glBindTexture(GL_TEXTURE_2D, ground_camera_texture_);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 240, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        ground_camera_texture_initialized_ = false;
        
        // Pre-allocate ground camera image buffer
        ground_camera_image_ = cv::Mat::zeros(240, 640, CV_8UC3);
        
        // Initialize update flag
        ground_camera_updated_ = false;
        
        RCLCPP_INFO(this->get_logger(), "Ground camera texture initialized (640x240)");
    }
    
    void updateCameraTextures() {
        std::string cam_names[] = {"frontCamera", "leftCamera", "rightCamera", "rearCamera"};
        
        for (int i = 0; i < 4; i++) {
            const std::string& cam_name = cam_names[i];
            
            // Only update if this specific camera has new data (lock-free check)
            if (!camera_updated_[cam_name]) {
                continue;
            }
            
            // Get the pre-undistorted image (no lock needed since it's already processed)
            cv::Mat& undistorted = undistorted_images_[cam_name];
            
            // Update OpenGL texture directly from undistorted image (no flip needed)
            glBindTexture(GL_TEXTURE_2D, camera_textures_[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, undistorted.cols, undistorted.rows, 0, 
                        GL_BGR, GL_UNSIGNED_BYTE, undistorted.data);
            
            texture_initialized_[i] = true;
            
            // Reset the update flag (lock-free)
            camera_updated_[cam_name] = false;
        }
    }
    
    void updateInsideCameraTexture() {
        // Only update if inside camera has new data (lock-free check)
        if (!inside_camera_updated_) {
            return;
        }
        
        // Update OpenGL texture directly from inside camera image
        glBindTexture(GL_TEXTURE_2D, inside_camera_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, inside_camera_image_.cols, inside_camera_image_.rows, 0, 
                    GL_BGR, GL_UNSIGNED_BYTE, inside_camera_image_.data);
        
        inside_camera_texture_initialized_ = true;
        
        // Reset the update flag (lock-free)
        inside_camera_updated_ = false;
    }
    
    void updateGroundCameraTexture() {
        // Only update if ground camera has new data (lock-free check)
        if (!ground_camera_updated_) {
            return;
        }
        
        // Update OpenGL texture directly from ground camera image
        glBindTexture(GL_TEXTURE_2D, ground_camera_texture_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, ground_camera_image_.cols, ground_camera_image_.rows, 0, 
                    GL_BGR, GL_UNSIGNED_BYTE, ground_camera_image_.data);
        
        ground_camera_texture_initialized_ = true;
        
        // Reset the update flag (lock-free)
        ground_camera_updated_ = false;
    }
    
    void toggleFullscreen() {
        if (is_fullscreen_) {
            // Return to windowed mode
            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            
            glfwSetWindowMonitor(window_, nullptr, windowed_x_, windowed_y_, 
                               windowed_width_, windowed_height_, 0);
            is_fullscreen_ = false;
            
            // Update viewport for windowed mode
            glViewport(0, 0, windowed_width_, windowed_height_);
            
            RCLCPP_INFO(this->get_logger(), "Switched to windowed mode: %dx%d at (%d,%d)", 
                       windowed_width_, windowed_height_, windowed_x_, windowed_y_);
        } else {
            // Switch to fullscreen mode
            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            
            glfwSetWindowMonitor(window_, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
            is_fullscreen_ = true;
            
            // Update viewport for fullscreen mode
            glViewport(0, 0, mode->width, mode->height);
            
            RCLCPP_INFO(this->get_logger(), "Switched to fullscreen mode: %dx%d", 
                       mode->width, mode->height);
        }
    }
    
    void handleFramebufferResize(int width, int height) {
        // Update viewport when window is resized (including maximize button)
        glViewport(0, 0, width, height);
        RCLCPP_DEBUG(this->get_logger(), "Viewport updated to %dx%d", width, height);
    }
    
    glm::mat4 getProjectionMatrix() {
        int width, height;
        glfwGetWindowSize(window_, &width, &height);
        return glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 100.0f);
    }
    
    void renderLoop() {
        try {
            initializeOpenGL();
            load3DModel();
            {
                std::lock_guard<std::mutex> lock(opengl_mutex_);
                opengl_initialized_ = true;
            }
            opengl_cv_.notify_one();
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize OpenGL: %s", e.what());
            should_exit_ = true;
            return;
        }
        glfwMakeContextCurrent(window_);
        
        // Optimized rendering loop for production
        const auto target_frame_time = std::chrono::milliseconds(16);  // ~60 FPS
        
        while (!should_exit_ && !glfwWindowShouldClose(window_)) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // Update auto-revolution (only if needed)
            static auto last_auto_rev_time = std::chrono::high_resolution_clock::now();
            auto current_time = std::chrono::high_resolution_clock::now();
            float delta_time = std::chrono::duration<float>(current_time - last_auto_rev_time).count();
            last_auto_rev_time = current_time;
            
            virtual_camera_.updateAutoRevolution(delta_time);
            
            // Only update textures if there are actual updates (selective update)
            if (needs_update_) {
                updateCameraTextures();
                updateInsideCameraTexture();
                updateGroundCameraTexture();
                needs_update_ = false;
            }
            
            glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            
            glUseProgram(shader_program_);
            
            glm::mat4 view = virtual_camera_.getViewMatrix();
            glm::mat4 projection = getProjectionMatrix();
            
            glUniformMatrix4fv(glGetUniformLocation(shader_program_, "view"), 1, GL_FALSE, glm::value_ptr(view));
            glUniformMatrix4fv(glGetUniformLocation(shader_program_, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
            
            renderBowlMesh();
            render3DModel();
            renderInsideCameraPlane();
            renderGroundCameraPlane();
            
            // Disable debug rendering for production efficiency
            // glDisable(GL_DEPTH_TEST);
            // renderAxes();
            // renderCameraFrustums();
            // glEnable(GL_DEPTH_TEST);
            
            glfwSwapBuffers(window_);
            glfwPollEvents();
            
            // Adaptive frame rate control for production efficiency
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start);
            
            if (frame_duration < target_frame_time) {
                std::this_thread::sleep_for(target_frame_time - frame_duration);
            }
            // print frame duration
            // RCLCPP_INFO(this->get_logger(), "Frame duration: %ld ms", frame_duration.count());
            // If frame took longer than target, don't sleep (maintain real-time performance)
        }
    }
    
    void renderBowlMesh() {
        glUseProgram(bowl_shader_program_);
        
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = virtual_camera_.getViewMatrix();
        glm::mat4 projection = getProjectionMatrix();
        
        glUniformMatrix4fv(glGetUniformLocation(bowl_shader_program_, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(bowl_shader_program_, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(bowl_shader_program_, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(glGetUniformLocation(bowl_shader_program_, "alpha"), 1.0f);
        
        for (int i = 0; i < 4; i++) {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, camera_textures_[i]);
        }
        
        int texture_units[4] = {0, 1, 2, 3};
        glUniform1iv(glGetUniformLocation(bowl_shader_program_, "cameraTextures"), 4, texture_units);
        
        int valid[4];
        for (int i = 0; i < 4; i++) { valid[i] = texture_initialized_[i] ? 1 : 0; }
        glUniform1iv(glGetUniformLocation(bowl_shader_program_, "textureValid"), 4, valid);
                
        glBindVertexArray(bowl_vao_);
        glDrawElements(GL_TRIANGLES, bowl_indices_.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glUseProgram(shader_program_);
    }
    
    void render3DModel() {
        if (model_meshes_.empty()) return;
        
        // Ensure we're using the standard shader
        glUseProgram(shader_program_);
        
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(0.0f, 0.0f, 0.0f));
        
        // Fix vehicle orientation to match the expected coordinate system:
        // - Front should point towards +X axis
        // - Right side should point towards -Y axis  
        // - Top should point towards +Z axis
        // If front points to -X and right points to +Z, we need to flip the model around Y axis
        model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        // model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        
        model = glm::scale(model, glm::vec3(1.0f, 1.0f, 1.0f));
        glUniformMatrix4fv(glGetUniformLocation(shader_program_, "model"), 1, GL_FALSE, glm::value_ptr(model));
        
        glUniform1f(glGetUniformLocation(shader_program_, "alpha"), 0.9f);
        
        // Set lighting uniforms - using the suggested lighting position
        glm::vec3 lightPos = glm::vec3(-10.0f, 10.0f, 10.0f);  // Light at [X=-10, Y=10, Z=10] in vehicle coordinates
        glm::vec3 viewPos = virtual_camera_.position;
        glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);  // White light
        glm::vec3 ambientColor = glm::vec3(0.3f, 0.3f, 0.3f);  // Ambient lighting
        
        glUniform3fv(glGetUniformLocation(shader_program_, "lightPos"), 1, glm::value_ptr(lightPos));
        glUniform3fv(glGetUniformLocation(shader_program_, "viewPos"), 1, glm::value_ptr(viewPos));
        glUniform3fv(glGetUniformLocation(shader_program_, "lightColor"), 1, glm::value_ptr(lightColor));
        glUniform3fv(glGetUniformLocation(shader_program_, "ambientColor"), 1, glm::value_ptr(ambientColor));
        
        // Render each mesh with its texture
        for (const auto& mesh : model_meshes_) {
            if (mesh.texture_id != 0) {
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, mesh.texture_id);
                glUniform1i(glGetUniformLocation(shader_program_, "texture1"), 0);
            }
            
            glBindVertexArray(mesh.vao);
            glDrawElements(GL_TRIANGLES, mesh.index_count, GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }
    }
    
    void renderInsideCameraPlane() {
        if (!inside_camera_texture_initialized_) {
            return; // Don't render if texture is not available
        }
        
        glUseProgram(inside_camera_shader_program_);
        
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = virtual_camera_.getViewMatrix();
        glm::mat4 projection = getProjectionMatrix();
        
        glUniformMatrix4fv(glGetUniformLocation(inside_camera_shader_program_, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(inside_camera_shader_program_, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(inside_camera_shader_program_, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(glGetUniformLocation(inside_camera_shader_program_, "alpha"), 1.0f);
        glUniform1i(glGetUniformLocation(inside_camera_shader_program_, "textureValid"), inside_camera_texture_initialized_ ? 1 : 0);
        
        // Bind the inside camera texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, inside_camera_texture_);
        glUniform1i(glGetUniformLocation(inside_camera_shader_program_, "insideCameraTexture"), 0);
        
        // Render the planar mesh
        glBindVertexArray(inside_camera_vao_);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    
    void renderGroundCameraPlane() {
        if (!ground_camera_texture_initialized_) {
            return; // Don't render if texture is not available
        }
        
        glUseProgram(ground_camera_shader_program_);
        
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = virtual_camera_.getViewMatrix();
        glm::mat4 projection = getProjectionMatrix();
        
        glUniformMatrix4fv(glGetUniformLocation(ground_camera_shader_program_, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(ground_camera_shader_program_, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(ground_camera_shader_program_, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform1f(glGetUniformLocation(ground_camera_shader_program_, "alpha"), 1.0f);
        glUniform1i(glGetUniformLocation(ground_camera_shader_program_, "textureValid"), ground_camera_texture_initialized_ ? 1 : 0);
        
        // Bind the ground camera texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ground_camera_texture_);
        glUniform1i(glGetUniformLocation(ground_camera_shader_program_, "groundCameraTexture"), 0);
        
        // Render the planar mesh
        glBindVertexArray(ground_camera_vao_);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
    
    void renderAxes() {
        glUseProgram(axes_shader_program_);
        
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = virtual_camera_.getViewMatrix();
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 
                                               (float)BEV_WIDTH / (float)BEV_HEIGHT, 
                                               0.1f, 100.0f);
        
        glUniformMatrix4fv(glGetUniformLocation(axes_shader_program_, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(axes_shader_program_, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(axes_shader_program_, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        
        glLineWidth(3.0f);
        
        glBindVertexArray(axes_vao_);
        glDrawElements(GL_LINES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        
        glLineWidth(1.0f);
    }

    void renderCameraFrustums() {
        glUseProgram(axes_shader_program_);

        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = virtual_camera_.getViewMatrix();
        glm::mat4 projection = getProjectionMatrix();
        
        glUniformMatrix4fv(glGetUniformLocation(axes_shader_program_, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(axes_shader_program_, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(axes_shader_program_, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        std::string cam_names[] = {"frontCamera", "leftCamera", "rightCamera", "rearCamera"};
        glm::vec3 colors[] = {glm::vec3(0.6f, 0.0f, 0.0f), glm::vec3(0.0f, 0.6f, 0.0f), glm::vec3(0.0f, 0.0f, 0.6f), glm::vec3(0.6f, 0.6f, 0.0f)};

        for (int i = 0; i < 4; i++) {
            auto& cam = cameras_[cam_names[i]];
            
            float fx = cam.K.at<double>(0, 0);
            float fy = cam.K.at<double>(1, 1);
            float cx = cam.K.at<double>(0, 2);
            float cy = cam.K.at<double>(1, 2);
            float w = cam.width;
            float h = cam.height;

            // Define frustum in camera space (looking down -Z, Y is up)
            float near_dist = 0.05f;
            float far_dist = 5.0f;

            std::vector<cv::Point2f> corners = {{0, 0}, {w, 0}, {w, h}, {0, h}};
            std::vector<glm::vec3> points_cam_space;

            for(const auto& p : corners) {
                points_cam_space.push_back(glm::vec3((p.x - cx) * near_dist / fx, -(p.y - cy) * near_dist / fy, -near_dist));
            }
            for(const auto& p : corners) {
                points_cam_space.push_back(glm::vec3((p.x - cx) * far_dist / fx, -(p.y - cy) * far_dist / fy, -far_dist));
            }
            
            glm::mat4 cam_to_world = glm::inverse(cam.extrinsic);
            std::vector<glm::vec3> points_world_space;
            for(const auto& p : points_cam_space) {
                points_world_space.push_back(glm::vec3(cam_to_world * glm::vec4(p, 1.0f)));
            }

            float vertices[24 * 6];
            unsigned int indices[] = {0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7};
            
            for(int j=0; j<24; ++j) {
                glm::vec3 point = points_world_space[indices[j]];
                vertices[j*6 + 0] = point.x;
                vertices[j*6 + 1] = point.y;
                vertices[j*6 + 2] = point.z;
                vertices[j*6 + 3] = colors[i].x;
                vertices[j*6 + 4] = colors[i].y;
                vertices[j*6 + 5] = colors[i].z;
            }
            
            GLuint temp_vao, temp_vbo;
            glGenVertexArrays(1, &temp_vao);
            glGenBuffers(1, &temp_vbo);

            glBindVertexArray(temp_vao);
            glBindBuffer(GL_ARRAY_BUFFER, temp_vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
            glEnableVertexAttribArray(1);

            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, 24);

            glDeleteVertexArrays(1, &temp_vao);
            glDeleteBuffers(1, &temp_vbo);
        }
    }

    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        SurroundVisionNode* node = static_cast<SurroundVisionNode*>(glfwGetWindowUserPointer(window));
        if (node && action == GLFW_PRESS) {
            node->handleKeyPress(key);
        }
    }

    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
        SurroundVisionNode* node = static_cast<SurroundVisionNode*>(glfwGetWindowUserPointer(window));
        if (node) {
            node->handleMouseButton(button, action, mods);
        }
    }

    static void mouseCursorCallback(GLFWwindow* window, double xpos, double ypos) {
        SurroundVisionNode* node = static_cast<SurroundVisionNode*>(glfwGetWindowUserPointer(window));
        if (node) {
            node->handleMouseCursor(xpos, ypos);
        }
    }

    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
        SurroundVisionNode* node = static_cast<SurroundVisionNode*>(glfwGetWindowUserPointer(window));
        if (node) {
            node->handleScroll(xoffset, yoffset);
        }
    }

    static void windowCloseCallback(GLFWwindow* window) {
        SurroundVisionNode* node = static_cast<SurroundVisionNode*>(glfwGetWindowUserPointer(window));
        if (node) {
            node->should_exit_ = true;
        }
    }
    
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
        SurroundVisionNode* node = static_cast<SurroundVisionNode*>(glfwGetWindowUserPointer(window));
        if (node) {
            node->handleFramebufferResize(width, height);
        }
    }

    void handleKeyPress(int key) {
        switch(key) {
            case GLFW_KEY_A: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.rotate(-1.0f, 0.0f, 0.0f); 
                break;
            case GLFW_KEY_D: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.rotate(1.0f, 0.0f, 0.0f); 
                break;
            case GLFW_KEY_W: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.rotate(0.0f, -1.0f, 0.0f); 
                break;
            case GLFW_KEY_S: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.rotate(0.0f, 1.0f, 0.0f); 
                break;
            case GLFW_KEY_Q: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.rotate(0.0f, 0.0f, -1.0f); 
                break;
            case GLFW_KEY_E: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.rotate(0.0f, 0.0f, 1.0f); 
                break;
            case GLFW_KEY_UP: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.move(1.0f, 0.0f, 0.0f); 
                break;
            case GLFW_KEY_DOWN: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.move(-1.0f, 0.0f, 0.0f); 
                break;
            case GLFW_KEY_LEFT: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.move(0.0f, 1.0f, 0.0f); 
                break;
            case GLFW_KEY_RIGHT: 
                virtual_camera_.stopAutoRevolution();
                virtual_camera_.move(0.0f, -1.0f, 0.0f); 
                break;
            case GLFW_KEY_X: 
                virtual_camera_.zoom(1.0f); 
                break;
            case GLFW_KEY_Z: 
                virtual_camera_.zoom(-1.0f); 
                break;
            case GLFW_KEY_SPACE: 
                virtual_camera_.startAutoRevolution(); 
                break;
            case GLFW_KEY_F: 
                toggleFullscreen(); 
                break;
        }
        needs_update_ = true;
    }

    void handleMouseButton(int button, int action, int mods) {
        std::lock_guard<std::mutex> lock(mouse_control_.mutex);
        
        if (action == GLFW_PRESS) {
            if(button == GLFW_MOUSE_BUTTON_LEFT) mouse_control_.left_pressed = true;
            if(button == GLFW_MOUSE_BUTTON_MIDDLE) mouse_control_.middle_pressed = true;
            if(button == GLFW_MOUSE_BUTTON_RIGHT) mouse_control_.right_pressed = true;
        } else if (action == GLFW_RELEASE) {
            if(button == GLFW_MOUSE_BUTTON_LEFT) mouse_control_.left_pressed = false;
            if(button == GLFW_MOUSE_BUTTON_MIDDLE) mouse_control_.middle_pressed = false;
            if(button == GLFW_MOUSE_BUTTON_RIGHT) mouse_control_.right_pressed = false;
        }
    }

    void handleMouseCursor(double xpos, double ypos) {
        std::lock_guard<std::mutex> lock(mouse_control_.mutex);
        
        if (mouse_control_.first_mouse) {
            mouse_control_.last_x = xpos;
            mouse_control_.last_y = ypos;
            mouse_control_.first_mouse = false;
            return;
        }

        double xoffset = xpos - mouse_control_.last_x;
        double yoffset = mouse_control_.last_y - ypos;
        mouse_control_.last_x = xpos;
        mouse_control_.last_y = ypos;

        if (mouse_control_.left_pressed) {
            virtual_camera_.stopAutoRevolution();
            virtual_camera_.rotate(-xoffset * mouse_control_.mouse_sensitivity, yoffset * mouse_control_.mouse_sensitivity);
        } else if (mouse_control_.middle_pressed) {
            virtual_camera_.stopAutoRevolution();
            float pan_x = -yoffset * mouse_control_.mouse_sensitivity * 0.1f;
            float pan_y = xoffset * mouse_control_.mouse_sensitivity * 0.1f;
            virtual_camera_.move(pan_x, pan_y, 0.0f);
        } else if (mouse_control_.right_pressed) {
            virtual_camera_.stopAutoRevolution();
            virtual_camera_.zoom(-yoffset * mouse_control_.zoom_sensitivity);
        }

        needs_update_ = true;
    }

    void handleScroll(double xoffset, double yoffset) {
        virtual_camera_.stopAutoRevolution();
        virtual_camera_.zoom(-yoffset * mouse_control_.zoom_sensitivity);
        needs_update_ = true;
    }

};

} // namespace surround_vision

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<surround_vision::SurroundVisionNode>();
        
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        
        std::thread spinner([&executor]() {
            executor.spin();
        });
        
        while (rclcpp::ok() && !node->shouldExit()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        executor.cancel();
        if (spinner.joinable()) {
            spinner.join();
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}

