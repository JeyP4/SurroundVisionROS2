#include "opencv2/opencv.hpp"
#include <iostream>

/**
 * @brief Generates remapping maps to convert a cylindrically warped image to a rectilinear image.
 *
 * This function computes the inverse transformation from a cylindrical projection to a 
 * standard rectilinear (perspective) projection. It automatically determines the output image size 
 * to ensure that there is no scaling along the vertical principal axis of the input image and that 
 * the entire source image is visible in the output. It calculates the necessary `mapx` and `mapy` 
 * for use with cv::remap, provides the new camera intrinsic matrix (K), and returns the calculated size.
 *
 * @param warper_radius The radius of the cylindrical warper used in the original cylindrical projection.
 * @param cylindrical_img_size The size of the input cylindrically warped image.
 * @param mapx_rect Output 32-bit floating-point map for the x-coordinates.
 * @param mapy_rect Output 32-bit floating-point map for the y-coordinates.
 * @param K_rect Output 3x3 double-precision camera intrinsic matrix for the new rectilinear view.
 */
cv::Size  createRectilinearMaps(
    const float warper_radius,
    const cv::Size& cylindrical_img_size,
    cv::Mat& mapx_rect,
    cv::Mat& mapy_rect,
    cv::Matx33d& K_rect,
    float scale=1.0f)
{    
    // The principal point of the cropped cylindrical image is its center.
    cv::Point2f cylindrical_principal_point(
        cylindrical_img_size.width / 2.0f,
        cylindrical_img_size.height / 2.0f
    );

    // 2. Determine output size and focal length to ensure no scaling on the vertical principal axis.
    // To achieve 1:1 pixel mapping (no scaling) on the vertical principal axis, 
    // the new focal length for the rectilinear projection must equal the cylindrical warper radius.
    double new_focal_length = warper_radius*scale;

    // The horizontal field of view (HFOV) is determined by the width of the cylindrical image and the radius.
    double hfov = static_cast<double>(cylindrical_img_size.width) / warper_radius;

    // Calculate the required rectilinear image width to fully contain the HFOV.
    int output_width = static_cast<int>(std::round(2.0 * new_focal_length * tan(hfov / 2.0)));
    
    // The top and bottom edges of the cylindrical image will appear curved in the rectilinear view.
    // To ensure these curved edges are not cropped, the output height must be larger than the input height.
    // The required height is calculated based on the y-coordinate of a corner point after unprojection.
    int output_height = static_cast<int>(std::round(static_cast<double>(cylindrical_img_size.height) / cos(hfov / 2.0) * scale));

    // Ensure dimensions are even, which can be a requirement for some video codecs.
    if (output_width % 2 != 0) output_width++;
    if (output_height % 2 != 0) output_height++;

    cv::Size output_rectilinear_size(output_width, output_height);

    // 3. Define the new camera intrinsic matrix for the rectilinear image.
    double cx_rect = (output_rectilinear_size.width - 1) / 2.0;
    double cy_rect = (output_rectilinear_size.height - 1) / 2.0;

    K_rect = cv::Matx33d(
        new_focal_length, 0,                cx_rect,
        0,                new_focal_length, cy_rect,
        0,                0,                1
    );

    // 4. Create the remapping matrices.
    mapx_rect.create(output_rectilinear_size, CV_32FC1);
    mapy_rect.create(output_rectilinear_size, CV_32FC1);

    // Iterate over each pixel of the destination (rectilinear) image
    for (int v = 0; v < output_rectilinear_size.height; ++v) {
        for (int u = 0; u < output_rectilinear_size.width; ++u) {
            
            // Convert the destination pixel (u, v) to a normalized 3D ray using the inverse of the new K matrix.
            double x_norm = (u - cx_rect) / new_focal_length;
            double y_norm = (v - cy_rect) / new_focal_length;
            
            // Apply the forward cylindrical projection equations to find the corresponding point on the cylindrical image plane.
            // theta is the horizontal angle, h is the normalized y-coordinate on the cylinder.
            double theta = atan(x_norm);
            double h = y_norm / sqrt(x_norm * x_norm + 1.0);
            
            // Convert from normalized cylindrical coordinates to pixel coordinates in the source cylindrical image.
            float u_cyl = static_cast<float>(warper_radius * theta + cylindrical_principal_point.x);
            float v_cyl = static_cast<float>(warper_radius * h + cylindrical_principal_point.y);

            // Store the source coordinates in the maps.
            mapx_rect.at<float>(v, u) = u_cyl;
            mapy_rect.at<float>(v, u) = v_cyl;
        }
    }

    // std::cout << "Successfully generated rectilinear maps." << std::endl;
    // std::cout << "New Rectilinear Camera Matrix (K_rect):" << std::endl << K_rect << std::endl;
    // std::cout << "Calculated Output Size: " << output_rectilinear_size.width << "x" << output_rectilinear_size.height << std::endl;

    return output_rectilinear_size;
}

