#ifndef IMAGE_DECODER_H
#define IMAGE_DECODER_H

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <string>

#ifdef USE_LIBJPEG_TURBO
#include <turbojpeg.h>
#endif

/**
 * @brief Modular image decoder class that supports both libturbojpeg and OpenCV fallback
 * 
 * This class provides efficient JPEG decoding using libturbojpeg when available,
 * with automatic fallback to OpenCV's imdecode when libturbojpeg is not available.
 * The interface remains the same regardless of the underlying implementation.
 */
class ImageDecoder {
public:
    /**
     * @brief Constructor - initializes the decoder
     * @param logger ROS2 logger for error reporting (optional)
     */
    ImageDecoder(rclcpp::Logger logger = rclcpp::get_logger("image_decoder"));
    
    /**
     * @brief Destructor - cleans up resources
     */
    ~ImageDecoder();
    
    /**
     * @brief Decode a compressed image message to OpenCV Mat
     * @param msg Compressed image message
     * @return Decoded image as cv::Mat, empty if decoding failed
     */
    cv::Mat decode(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
    
    /**
     * @brief Check if libturbojpeg is available and being used
     * @return true if using libturbojpeg, false if using OpenCV fallback
     */
    bool isUsingTurboJpeg() const { return turbojpeg_available_; }
    
    /**
     * @brief Get the format of the compressed image
     * @param msg Compressed image message
     * @return Format string (e.g., "jpeg", "png", etc.)
     */
    std::string getImageFormat(const sensor_msgs::msg::CompressedImage::SharedPtr msg) const;

private:
    rclcpp::Logger logger_;
    bool turbojpeg_available_;
    
#ifdef USE_LIBJPEG_TURBO
    tjhandle tjpeg_handle_;
    
    /**
     * @brief Initialize libturbojpeg decompressor
     * @return true if successful, false otherwise
     */
    bool initTurboJpeg();
    
    /**
     * @brief Cleanup libturbojpeg resources
     */
    void cleanupTurboJpeg();
    
    /**
     * @brief Decode JPEG using libturbojpeg
     * @param msg Compressed image message
     * @return Decoded image as cv::Mat, empty if decoding failed
     */
    cv::Mat decodeWithTurboJpeg(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
#endif
    
    /**
     * @brief Decode image using OpenCV fallback
     * @param msg Compressed image message
     * @return Decoded image as cv::Mat, empty if decoding failed
     */
    cv::Mat decodeWithOpenCV(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
};

#endif // IMAGE_DECODER_H
