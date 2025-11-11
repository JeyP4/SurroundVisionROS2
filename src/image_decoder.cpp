#include "image_decoder.h"
#include <rclcpp/rclcpp.hpp>
#include <algorithm>
#include <cctype>

ImageDecoder::ImageDecoder(rclcpp::Logger logger) 
    : logger_(logger), turbojpeg_available_(false) {
#ifdef USE_LIBJPEG_TURBO
    turbojpeg_available_ = initTurboJpeg();
    if (turbojpeg_available_) {
        RCLCPP_INFO(logger_, "ImageDecoder: Using libturbojpeg for efficient JPEG decoding");
    } else {
        RCLCPP_WARN(logger_, "ImageDecoder: libturbojpeg initialization failed, using OpenCV fallback");
    }
#else
    RCLCPP_INFO(logger_, "ImageDecoder: libturbojpeg not available, using OpenCV fallback");
#endif
}

ImageDecoder::~ImageDecoder() {
#ifdef USE_LIBJPEG_TURBO
    cleanupTurboJpeg();
#endif
}

cv::Mat ImageDecoder::decode(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    if (!msg || msg->data.empty()) {
        RCLCPP_WARN(logger_, "%s: Empty or null compressed image message", msg->header.frame_id.c_str());
        return cv::Mat();
    }
    
    // Check if it's a JPEG image and we have turbojpeg available
    std::string format = getImageFormat(msg);
    // print the format
    // RCLCPP_INFO(logger_, "ImageDecoder: Image format: %s", format.c_str());
#ifdef USE_LIBJPEG_TURBO
    if (turbojpeg_available_ && (format.find("jpeg") != std::string::npos || format.find("jpg") != std::string::npos)) {
        return decodeWithTurboJpeg(msg);
    }
#endif
    
    // Fallback to OpenCV for non-JPEG or when turbojpeg is not available
    return decodeWithOpenCV(msg);
}

std::string ImageDecoder::getImageFormat(const sensor_msgs::msg::CompressedImage::SharedPtr msg) const {
    if (!msg || msg->format.empty()) {
        return "unknown";
    }
    
    // Convert to lowercase for comparison
    std::string format = msg->format;
    std::transform(format.begin(), format.end(), format.begin(), ::tolower);
    
    return format;
}

cv::Mat ImageDecoder::decodeWithOpenCV(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    try {
        cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (image.empty()) {
            RCLCPP_WARN(logger_, "%s: OpenCV failed to decode image", msg->header.frame_id.c_str());
        }
        return image;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "%s: OpenCV decode error: %s", msg->header.frame_id.c_str(), e.what());
        return cv::Mat();
    }
}

#ifdef USE_LIBJPEG_TURBO
bool ImageDecoder::initTurboJpeg() {
    try {
        tjpeg_handle_ = tjInitDecompress();
        if (!tjpeg_handle_) {
            RCLCPP_ERROR(logger_, "ImageDecoder: Failed to initialize libturbojpeg decompressor");
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "ImageDecoder: Exception during turbojpeg initialization: %s", e.what());
        return false;
    }
}

void ImageDecoder::cleanupTurboJpeg() {
    if (tjpeg_handle_) {
        tjDestroy(tjpeg_handle_);
        tjpeg_handle_ = nullptr;
    }
}

cv::Mat ImageDecoder::decodeWithTurboJpeg(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
    try {
        int width, height, subsamp, colorspace;
        
        // Get image info
        int result = tjDecompressHeader3(tjpeg_handle_, 
                                       msg->data.data(), 
                                       msg->data.size(),
                                       &width, &height, &subsamp, &colorspace);
        
        if (result != 0) {
            RCLCPP_WARN(logger_, "%s: libturbojpeg failed to get header: %s", msg->header.frame_id.c_str(), tjGetErrorStr());
            return decodeWithOpenCV(msg); // Fallback to OpenCV
        }
        
        // Create output image
        cv::Mat image(height, width, CV_8UC3);
        
        // Decompress to BGR format (OpenCV standard)
        result = tjDecompress2(tjpeg_handle_,
                             msg->data.data(),
                             msg->data.size(),
                             image.data,
                             width, 0, height,
                             TJPF_BGR, TJFLAG_FASTDCT);
        
        if (result != 0) {
            RCLCPP_WARN(logger_, "%s: libturbojpeg decompression failed: %s", msg->header.frame_id.c_str(), tjGetErrorStr());
            return decodeWithOpenCV(msg); // Fallback to OpenCV
        }
        
        return image;
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "%s: Exception during turbojpeg decode: %s", msg->header.frame_id.c_str(), e.what());
        return decodeWithOpenCV(msg); // Fallback to OpenCV
    }
}
#endif
