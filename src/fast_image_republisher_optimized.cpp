#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <mutex>

// GPU acceleration permanently disabled to avoid CUDA version conflicts

// #define USE_LIBJPEG_TURBO  // Already defined in CMakeLists.txt
// libjpeg-turbo includes
#ifdef USE_LIBJPEG_TURBO
#include <turbojpeg.h>
#endif

class FastImageRepublisher : public rclcpp::Node
{
public:
    FastImageRepublisher() : Node("fast_image_republisher")
    {
        // Declare parameters
        this->declare_parameter("camera_names", std::vector<std::string>{"front", "left", "rear", "right"});
        this->declare_parameter("use_gpu", false);  // GPU permanently disabled
        this->declare_parameter("use_libjpeg_turbo", true);
        this->declare_parameter("enable_timing", true);
        this->declare_parameter("reverse_color_channels", true);
        this->declare_parameter("qos_depth", 1);
        this->declare_parameter("max_image_width", 1920);
        this->declare_parameter("max_image_height", 1080);
        
        // Get parameters
        camera_names_ = this->get_parameter("camera_names").as_string_array();
        use_gpu_ = this->get_parameter("use_gpu").as_bool();
        use_libjpeg_turbo_ = this->get_parameter("use_libjpeg_turbo").as_bool();
        enable_timing_ = this->get_parameter("enable_timing").as_bool();
        reverse_color_channels_ = this->get_parameter("reverse_color_channels").as_bool();
        qos_depth_ = this->get_parameter("qos_depth").as_int();
        max_image_width_ = this->get_parameter("max_image_width").as_int();
        max_image_height_ = this->get_parameter("max_image_height").as_int();
        
        RCLCPP_INFO(this->get_logger(), "Setting up fast republishers for cameras: %s", 
                    join(camera_names_, ", ").c_str());
        
        // GPU initialization disabled
        
        // Initialize libjpeg-turbo if available
        if (use_libjpeg_turbo_) {
            initLibJpegTurbo();
        }
        
        // Setup QoS
        auto qos = rclcpp::QoS(qos_depth_)
            .reliability(rclcpp::ReliabilityPolicy::BestEffort)
            .durability(rclcpp::DurabilityPolicy::Volatile)
            .history(rclcpp::HistoryPolicy::KeepLast);
        
        // Setup publishers and subscribers
        setupPublishersAndSubscribers(qos);
        
        RCLCPP_INFO(this->get_logger(), "Fast Image Republisher initialized successfully");
    }
    
    ~FastImageRepublisher()
    {
        if (use_libjpeg_turbo_) {
            cleanupLibJpegTurbo();
        }
    }

private:
    // GPU functions removed - using CPU-only processing
    
    void initLibJpegTurbo()
    {
#ifdef USE_LIBJPEG_TURBO
        try {
            // Create turbojpeg handle
            tjpeg_handle_ = tjInitDecompress();
            if (!tjpeg_handle_) {
                RCLCPP_WARN(this->get_logger(), "Failed to initialize libjpeg-turbo decompressor");
                use_libjpeg_turbo_ = false;
                return;
            }
            
            RCLCPP_INFO(this->get_logger(), "libjpeg-turbo initialized successfully");
            
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "libjpeg-turbo initialization failed: %s", e.what());
            use_libjpeg_turbo_ = false;
        }
#else
        RCLCPP_WARN(this->get_logger(), "libjpeg-turbo support not compiled in");
        use_libjpeg_turbo_ = false;
#endif
    }
    
    void cleanupLibJpegTurbo()
    {
#ifdef USE_LIBJPEG_TURBO
        if (tjpeg_handle_) {
            tjDestroy(tjpeg_handle_);
            tjpeg_handle_ = nullptr;
        }
#endif
    }
    
    void setupPublishersAndSubscribers(const rclcpp::QoS& qos)
    {
        for (const auto& camera_name : camera_names_) {
            // Create publisher
            std::string pub_topic = "/" + camera_name + "Camera/image";
            auto publisher = this->create_publisher<sensor_msgs::msg::Image>(pub_topic, qos);
            publishers_[camera_name] = publisher;
            
            // Create subscriber with callback
            std::string sub_topic = "/" + camera_name + "Camera/v4l2/compressed";
            auto callback = [this, camera_name](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                this->imageCallback(msg, camera_name);
            };
            
            auto subscription = this->create_subscription<sensor_msgs::msg::CompressedImage>(
                sub_topic, qos, callback);
            subscriptions_.push_back(subscription);
            
            RCLCPP_INFO(this->get_logger(), "Created publisher/subscriber pair for %s", camera_name.c_str());
        }
    }
    
    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg, const std::string& camera_name)
    {

        if (publishers_[camera_name]->get_subscription_count() == 0) {
            RCLCPP_WARN(this->get_logger(), "No subscribers for %s, skipping image callback", camera_name.c_str());
            return;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            cv::Mat decoded_image;
            
            if (use_libjpeg_turbo_) {
                decoded_image = decodeLibJpegTurbo(msg);
            } else {
                decoded_image = decodeOpenCV(msg);
            }
            
            if (decoded_image.empty()) {
                RCLCPP_WARN(this->get_logger(), "Failed to decode image from %s", camera_name.c_str());
                return;
            }
            
            // Color channel conversion if needed
            if (reverse_color_channels_) {
                cv::cvtColor(decoded_image, decoded_image, cv::COLOR_BGR2RGB);
            }
            
            // Create and publish ROS message
            auto ros_image = cv_bridge::CvImage(msg->header, "bgr8", decoded_image).toImageMsg();
            publishers_[camera_name]->publish(*ros_image);
            
            // Log timing if enabled
            if (enable_timing_) {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                RCLCPP_INFO(this->get_logger(), "[%s] Processed in %.2f ms", 
                           camera_name.c_str(), duration.count() / 1000.0);
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error processing image from %s: %s", 
                        camera_name.c_str(), e.what());
        }
    }
    
    // decodeGPU function removed - using CPU-only processing
    
    cv::Mat decodeLibJpegTurbo(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
#ifdef USE_LIBJPEG_TURBO
        try {
            int width, height, subsamp, colorspace;
            
            // Get image info
            int result = tjDecompressHeader3(tjpeg_handle_, 
                                           msg->data.data(), 
                                           msg->data.size(), 
                                           &width, &height, &subsamp, &colorspace);
            if (result != 0) {
                RCLCPP_ERROR(this->get_logger(), "Failed to get JPEG header: %s", tjGetErrorStr2(tjpeg_handle_));
                return cv::Mat();
            }
            
            // Create output buffer
            cv::Mat result_mat(height, width, CV_8UC3);
            
            // Decompress JPEG
            result = tjDecompress2(tjpeg_handle_, 
                                  msg->data.data(), 
                                  msg->data.size(), 
                                  result_mat.data, 
                                  width, 
                                  0, // pitch
                                  height, 
                                  TJPF_BGR, 
                                  TJFLAG_FASTDCT);
            if (result != 0) {
                RCLCPP_ERROR(this->get_logger(), "Failed to decompress JPEG: %s", tjGetErrorStr2(tjpeg_handle_));
                return cv::Mat();
            }
            
            return result_mat;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "libjpeg-turbo decode error: %s", e.what());
            return cv::Mat();
        }
#else
        RCLCPP_WARN(this->get_logger(), "libjpeg-turbo decode requested but not available");
        return cv::Mat();
#endif
    }
    
    cv::Mat decodeOpenCV(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        try {
            // Use OpenCV's optimized imdecode
            std::vector<uchar> data(msg->data.begin(), msg->data.end());
            cv::Mat result = cv::imdecode(data, cv::IMREAD_COLOR);
            
            if (result.empty()) {
                RCLCPP_ERROR(this->get_logger(), "OpenCV imdecode failed");
                return cv::Mat();
            }
            
            return result;
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "OpenCV decode error: %s", e.what());
            return cv::Mat();
        }
    }
    
    std::string join(const std::vector<std::string>& vec, const std::string& delimiter)
    {
        std::string result;
        for (size_t i = 0; i < vec.size(); ++i) {
            if (i > 0) result += delimiter;
            result += vec[i];
        }
        return result;
    }
    
    // Member variables
    std::vector<std::string> camera_names_;
    std::unordered_map<std::string, rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr> publishers_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> subscriptions_;
    
    bool use_gpu_;
    bool use_libjpeg_turbo_;
    bool enable_timing_;
    bool reverse_color_channels_;
    int qos_depth_;
    int max_image_width_;
    int max_image_height_;
    
    // GPU-related variables removed
    
    // libjpeg-turbo variables
#ifdef USE_LIBJPEG_TURBO
    tjhandle tjpeg_handle_ = nullptr;
#endif
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<FastImageRepublisher>();
        
        // Use MultiThreadedExecutor for better performance
        rclcpp::executors::MultiThreadedExecutor executor;
        executor.add_node(node);
        
        RCLCPP_INFO(node->get_logger(), "Fast Image Republisher node is running...");
        
        executor.spin();
        
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("fast_image_republisher"), "Fatal error: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
