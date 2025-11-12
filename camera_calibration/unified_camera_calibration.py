#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import yaml
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
import math
import os
import time
import threading


class UnifiedCameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('unified_camera_calibration_node')
        
        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        
        # Initialize marker publisher for FOV visualization
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.marker_publisher = self.create_publisher(MarkerArray, '/camera_fov_markers', qos_profile)
        
        # Initialize marker publisher for calibration debug visualization
        self.points_marker_publisher = self.create_publisher(MarkerArray, '/calibration_points', qos_profile)

        # Camera names
        self.camera_names = ['frontCamera', 'leftCamera', 'rightCamera', 'rearCamera']
        
        # Paths
        self.param_file = '/home/jai/ros2_ws/src/surround_vision/config/camerasParam.yaml'
        self.output_file = '/home/jai/ros2_ws/src/surround_vision/config/camerasParam_extrinsic.yaml'
        self.calibration_points_file = '/home/jai/ros2_ws/src/surround_vision/config/calibration_points.yaml'
        self.scripts_dir = '/home/jai/ros2_ws/src/surround_vision/camera_calibration'
        
        # Load calibration points
        self.world_points, self.image_points, self.image_files = self.load_calibration_points()
        
        # Validate calibration points
        self.validate_calibration_points()
        
        # Load existing camera parameters
        self.camera_params = self.load_camera_parameters()
        
        # Perform calibration
        self.calibrate_cameras()
        
        # Save updated parameters
        self.save_updated_parameters()
        
        # Publish static transforms
        time.sleep(2)  # wait to ensure ros2 is ready
        self.publish_static_transforms()
        
        # Create and publish FOV markers
        self.create_fov_markers()
        
        # Create visualization overlays
        self.create_visualization_overlays()

        # Create and publish calibration points markers
        self.create_world_point_markers()
        
        self.get_logger().info("Camera calibration completed successfully!")
        
        # Start publishing loop in a separate thread
        # self.running = True
        # self.publish_thread = threading.Thread(target=self.publish_loop, daemon=True)
        # self.publish_thread.start()
    
    def load_calibration_points(self):
        """Load calibration points from YAML file"""
        self.get_logger().info(f"Loading calibration points from {self.calibration_points_file}")
        
        with open(self.calibration_points_file, 'r') as file:
            config = yaml.safe_load(file)
        
        world_points = {}
        image_points = {}
        image_files = {}
        
        # Load world points
        for point_set, points in config['world_points'].items():
            world_points[point_set] = np.array(points, dtype=np.float32)
        
        # Load image points
        for camera_name, point_sets in config['image_points'].items():
            image_points[camera_name] = {}
            for point_set, points in point_sets.items():
                image_points[camera_name][point_set] = np.array(points, dtype=np.float32)
        
        # Load image file paths
        if 'image_files' in config:
            image_files = config['image_files']
        
        return world_points, image_points, image_files
    
    def create_world_point_markers(self):
        """Create markers for world points visualization"""
        marker_array = MarkerArray()
        
        # Colors for different point sets
        colors = {
            'FL': [1.0, 1.0, 0.0, 1.0],  # yellow
            'FR': [135./255, 61./255, 204/255, 1.0],  # violet
            'RL': [135./255, 204./255, 61./255, 1.0],  # green
            'RR': [61./255, 204./255, 194./255, 1.0]   # cyan
        }
        
        marker_id = 0
        
        for point_set, points in self.world_points.items():
            for i, point in enumerate(points):
                # Create sphere marker for world point
                marker = Marker()
                marker.header.frame_id = 'vehicle'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.id = marker_id
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # Set position
                marker.pose.position.x = float(point[0])
                marker.pose.position.y = float(point[1])
                marker.pose.position.z = float(point[2])
                marker.pose.orientation.w = 1.0
                
                # Set scale
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                
                # Set color
                marker.color = ColorRGBA(r=colors[point_set][0], g=colors[point_set][1], 
                                       b=colors[point_set][2], a=colors[point_set][3])
                
                marker_array.markers.append(marker)
                marker_id += 1
                
                # Create text marker for point label
                text_marker = Marker()
                text_marker.header.frame_id = 'vehicle'
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.id = marker_id
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = float(point[0])
                text_marker.pose.position.y = float(point[1])
                text_marker.pose.position.z = float(point[2] + 0.2)
                text_marker.scale.z = 0.1
                text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                text_marker.text = f"{point_set}_{i+1}"
                
                marker_array.markers.append(text_marker)
                marker_id += 1
        
        # # Create vehicle reference frame marker, in case vehicle model is not available
        # vehicle_marker = Marker()
        # vehicle_marker.header.frame_id = 'vehicle'
        # vehicle_marker.header.stamp = self.get_clock().now().to_msg()
        # vehicle_marker.id = marker_id
        # vehicle_marker.type = Marker.CUBE
        # vehicle_marker.action = Marker.ADD
        # vehicle_marker.pose.position.x = 0.0
        # vehicle_marker.pose.position.y = 0.0
        # vehicle_marker.pose.position.z = 0.05
        # vehicle_marker.pose.orientation.w = 1.0
        # vehicle_marker.scale.x = 3.0  # Vehicle length
        # vehicle_marker.scale.y = 2.0  # Vehicle width
        # vehicle_marker.scale.z = 0.1  # Vehicle height
        # vehicle_marker.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.3)
        # marker_array.markers.append(vehicle_marker)
        
        # Publish markers
        self.points_marker_publisher.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} world point markers")
        
    
    def validate_calibration_points(self):
        """Validate that calibration points meet minimum requirements"""
        self.get_logger().info("Validating calibration points...")
        
        point_sets = ['FL', 'FR', 'RL', 'RR']
        
        for point_set in point_sets:
            if point_set not in self.world_points:
                raise ValueError(f"Missing world points for set {point_set}")
            
            num_points = len(self.world_points[point_set])
            if num_points < 3:
                raise ValueError(f"Point set {point_set} has only {num_points} points. Minimum 3 required.")
            
            self.get_logger().info(f"  {point_set}: {num_points} world points")
            
            # Check corresponding image points for each camera
            if point_set == 'FL':
                cameras = ['frontCamera', 'leftCamera']
            elif point_set == 'FR':
                cameras = ['frontCamera', 'rightCamera']
            elif point_set == 'RL':
                cameras = ['leftCamera', 'rearCamera']
            elif point_set == 'RR':
                cameras = ['rightCamera', 'rearCamera']
            
            for camera_name in cameras:
                if camera_name not in self.image_points:
                    raise ValueError(f"Missing image points for {camera_name}")
                if point_set not in self.image_points[camera_name]:
                    raise ValueError(f"Missing image points for {camera_name} point set {point_set}")
                
                num_img_points = len(self.image_points[camera_name][point_set])
                if num_img_points != num_points:
                    raise ValueError(
                        f"Mismatch: {point_set} has {num_points} world points but "
                        f"{camera_name} has {num_img_points} image points"
                    )
        
        self.get_logger().info("Calibration points validation passed!")
    
    def load_camera_parameters(self):
        """Load camera intrinsic parameters from camerasParam.yaml"""
        self.get_logger().info(f"Loading camera parameters from {self.param_file}")
        
        with open(self.param_file, 'r') as file:
            params = yaml.safe_load(file)
        
        return params
    
    def calibrate_cameras(self):
        """Calibrate extrinsic parameters for each camera"""
        self.get_logger().info("Starting camera calibration...")
        
        for camera_name in self.camera_names:
            self.get_logger().info(f"Calibrating {camera_name}...")
            
            # Get intrinsic parameters
            intrinsic_matrix = np.array(
                self.camera_params[camera_name]['ros__parameters']['intrinsic_matrix']
            ).reshape(3, 3)
            distortion_coeffs = np.array(
                self.camera_params[camera_name]['ros__parameters']['distortion_coefficients']
            )
            
            # Collect all world and image points for this camera
            world_points_list = []
            image_points_list = []
            
            # Determine which point sets this camera uses
            if camera_name == 'frontCamera':
                point_sets = ['FL', 'FR']
            elif camera_name == 'leftCamera':
                point_sets = ['FL', 'RL']
            elif camera_name == 'rightCamera':
                point_sets = ['FR', 'RR']
            elif camera_name == 'rearCamera':
                point_sets = ['RL', 'RR']
            
            for point_set in point_sets:
                world_points_list.extend(self.world_points[point_set])
                image_points_list.extend(self.image_points[camera_name][point_set])
            
            world_points_array = np.array(world_points_list, dtype=np.float32)
            image_points_array = np.array(image_points_list, dtype=np.float32)
            
            # Ensure image points are in the correct format for fisheye.solvePnP
            image_points_array = image_points_array.reshape(-1, 1, 2).astype(np.float32)
            world_points_array = world_points_array.reshape(-1, 1, 3).astype(np.float32)
            
            # Convert distortion coefficients to fisheye format
            fisheye_dist_coeffs = self.convert_to_fisheye_distortion(distortion_coeffs)
            
            self.get_logger().info(
                f"  Using {len(world_points_list)} points for calibration"
            )
            self.get_logger().info(
                f"  Fisheye distortion coefficients: {fisheye_dist_coeffs}"
            )
            
            # Solve PnP to get rotation and translation vectors for fisheye camera
            success, rvec, tvec = cv2.fisheye.solvePnP(
                world_points_array,
                image_points_array,
                intrinsic_matrix,
                fisheye_dist_coeffs,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Create 4x4 transformation matrix (world to camera)
                # Note: OpenCV returns camera to world, so we need to invert
                R_cam_to_world = rotation_matrix
                t_cam_to_world = tvec.flatten()
                
                # Convert to world to camera transformation
                R_world_to_cam = R_cam_to_world.T
                t_world_to_cam = -R_cam_to_world.T @ t_cam_to_world
                
                # Create 4x4 homogeneous transformation matrix
                transformation_matrix = np.eye(4)
                transformation_matrix[:3, :3] = R_world_to_cam
                transformation_matrix[:3, 3] = t_world_to_cam
                
                # Store the transformation matrix in row-major format
                self.camera_params[camera_name]['ros__parameters']['transformation_matrix'] = (
                    transformation_matrix.flatten().tolist()
                )
                
                self.get_logger().info(f"Successfully calibrated {camera_name}")
                self.get_logger().info(f"  Translation: {t_world_to_cam}")
                self.get_logger().info(f"  Rotation matrix:\n{R_world_to_cam}")
            else:
                self.get_logger().error(f"Failed to calibrate {camera_name}")
                raise RuntimeError(f"Calibration failed for {camera_name}")
    
    def save_updated_parameters(self):
        """Save updated camera parameters to camerasParam_extrinsic.yaml"""
        self.get_logger().info(f"Saving updated parameters to {self.output_file}")
        
        # Create the YAML content with proper formatting
        yaml_content = self.create_formatted_yaml()
        
        with open(self.output_file, 'w') as file:
            file.write(yaml_content)
        
        self.get_logger().info(f"Updated parameters saved to {self.output_file}")
    
    def create_formatted_yaml(self):
        """Create YAML content with proper formatting to match the original format"""
        yaml_lines = []
        
        # Add header comments
        yaml_lines.extend([
            "# Standardized Camera Configuration",
            "# Reference Frame: Rear-axle center projected on ground",
            "# Vehicle/world Coordinate System: X-forward, Y-leftward, Z-upward",
            "# Camera Coordinate System: X-rightward, Y-downward, Z-forward",
            ""
        ])
        
        # Process each camera
        for camera_name in self.camera_names:
            params = self.camera_params[camera_name]['ros__parameters']
            
            yaml_lines.append(f"{camera_name}:")
            yaml_lines.append("  ros__parameters:")
            
            # Transformation matrix with comment
            yaml_lines.append("    # 4x4 homogeneous transformation matrix (vehicle to camera)")
            yaml_lines.append(
                "    # Row-major format: [R11, R12, R13, Tx, R21, R22, R23, Ty, R31, R32, R33, Tz, 0.0, 0.0, 0.0, 1.0]"
            )
            transform_matrix_str = "[" + ", ".join(
                [f"{val:.10g}" for val in params['transformation_matrix']]
            ) + "]"
            yaml_lines.append(f"    transformation_matrix: {transform_matrix_str}")
            yaml_lines.append("")
            
            # Intrinsic matrix with comment
            yaml_lines.append("    # Intrinsic parameters")
            intrinsic_matrix_str = "[" + ", ".join(
                [f"{val:.10g}" for val in params['intrinsic_matrix']]
            ) + "]"
            yaml_lines.append(f"    intrinsic_matrix: {intrinsic_matrix_str}")
            
            # Distortion coefficients with comment
            distortion_str = "[" + ", ".join(
                [f"{val:.10g}" for val in params['distortion_coefficients']]
            ) + "]"
            yaml_lines.append(
                f"    distortion_coefficients: {distortion_str} # fisheye distortion coefficients"
            )
            yaml_lines.append("")
            
            # Image properties with comment
            yaml_lines.append("    # Image properties")
            yaml_lines.append(f"    image_width: {params['image_width']}")
            yaml_lines.append(f"    image_height: {params['image_height']}")
            yaml_lines.append(f"    field_of_view_horizontal: {params['field_of_view_horizontal']}")
            yaml_lines.append("")
        
        # Add other sections from original file (frontCamera_dec, rearCamera_dec, lidar, vehicle)
        for section_name in ['frontCamera_dec', 'rearCamera_dec', 'lidar', 'vehicle']:
            if section_name in self.camera_params:
                params = self.camera_params[section_name]['ros__parameters']
                
                yaml_lines.append(f"{section_name}:")
                yaml_lines.append("  ros__parameters:")
                
                if section_name in ['frontCamera_dec', 'rearCamera_dec']:
                    # Same format as regular cameras
                    yaml_lines.append("    # 4x4 homogeneous transformation matrix (vehicle to camera)")
                    yaml_lines.append(
                        "    # Row-major format: [R11, R12, R13, Tx, R21, R22, R23, Ty, R31, R32, R33, Tz, 0.0, 0.0, 0.0, 1.0]"
                    )
                    transform_matrix_str = "[" + ", ".join(
                        [f"{val:.10g}" for val in params['transformation_matrix']]
                    ) + "]"
                    yaml_lines.append(f"    transformation_matrix: {transform_matrix_str}")
                    yaml_lines.append("")
                    
                    yaml_lines.append("    # Intrinsic parameters")
                    intrinsic_matrix_str = "[" + ", ".join(
                        [f"{val:.10g}" for val in params['intrinsic_matrix']]
                    ) + "]"
                    yaml_lines.append(f"    intrinsic_matrix: {intrinsic_matrix_str}")
                    
                    distortion_str = "[" + ", ".join(
                        [f"{val:.10g}" for val in params['distortion_coefficients']]
                    ) + "]"
                    yaml_lines.append(f"    distortion_coefficients: {distortion_str}")
                    yaml_lines.append("")
                    
                    yaml_lines.append("    # Image properties")
                    yaml_lines.append(f"    image_width: {params['image_width']}")
                    yaml_lines.append(f"    image_height: {params['image_height']}")
                    yaml_lines.append(f"    field_of_view_horizontal: {params['field_of_view_horizontal']}")
                    yaml_lines.append("")
                    
                    yaml_lines.append("    # Processing parameters")
                    yaml_lines.append(f"    width_crop: {params['width_crop']}")
                    yaml_lines.append(f"    stretch_factor: {params['stretch_factor']}")
                    yaml_lines.append("")
                
                elif section_name == 'lidar':
                    # Lidar section
                    transform_matrix_str = "[" + ", ".join(
                        [f"{val:.10g}" for val in params['transformation_matrix']]
                    ) + "]"
                    yaml_lines.append(f"    transformation_matrix: {transform_matrix_str}")
                    yaml_lines.append("")
                
                elif section_name == 'vehicle':
                    # Vehicle section
                    yaml_lines.append(f"    width: {params['width']}")
                    yaml_lines.append(f"    length: {params['length']}")
                    yaml_lines.append(
                        f"    rear_axle_to_front: {params['rear_axle_to_front']}  # Distance from rear axle to front of vehicle"
                    )
                    yaml_lines.append("    coordinate_systems:")
                    yaml_lines.append("      vehicle:")
                    yaml_lines.append('        origin: "rear_axle_center_ground"')
                    yaml_lines.append("        axes:")
                    yaml_lines.append('          x: "forward"')
                    yaml_lines.append('          y: "leftward"')
                    yaml_lines.append('          z: "upward"')
                    yaml_lines.append("      camera:")
                    yaml_lines.append("        axes:")
                    yaml_lines.append('          x: "rightward"')
                    yaml_lines.append('          y: "downward"')
                    yaml_lines.append('          z: "forward"')
                    yaml_lines.append("")
        
        return "\n".join(yaml_lines)
    
    def publish_static_transforms(self):
        """Publish static transforms for all cameras"""
        self.get_logger().info("Publishing static transforms...")
        
        transforms = []
        
        for camera_name in self.camera_names:
            # Get transformation matrix
            transform_matrix = np.array(
                self.camera_params[camera_name]['ros__parameters']['transformation_matrix']
            ).reshape(4, 4)
            
            # Create transform from vehicle frame to camera frame
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = 'vehicle'
            transform.child_frame_id = camera_name
            
            # Extract rotation and translation
            rotation_matrix = transform_matrix[:3, :3]
            translation = transform_matrix[:3, 3]
            
            # Convert rotation matrix to quaternion
            quaternion = self.rotation_matrix_to_quaternion(rotation_matrix)
            
            # Set translation
            transform.transform.translation.x = float(translation[0])
            transform.transform.translation.y = float(translation[1])
            transform.transform.translation.z = float(translation[2])
            
            # Set rotation
            transform.transform.rotation.x = float(quaternion[0])
            transform.transform.rotation.y = float(quaternion[1])
            transform.transform.rotation.z = float(quaternion[2])
            transform.transform.rotation.w = float(quaternion[3])
            
            transforms.append(transform)
        
        # Broadcast all transforms
        self.tf_broadcaster.sendTransform(transforms)
        self.get_logger().info("Static transforms published")
    
    def create_fov_markers(self):
        """Create and publish camera FOV frustums as markers"""
        self.get_logger().info("Creating FOV markers...")
        
        marker_array = MarkerArray()
        
        for i, camera_name in enumerate(self.camera_names):
            # Get camera parameters
            intrinsic_matrix = np.array(
                self.camera_params[camera_name]['ros__parameters']['intrinsic_matrix']
            ).reshape(3, 3)
            # Get camera distortion coefficients
            distortion_coeffs = np.array(
                self.camera_params[camera_name]['ros__parameters']['distortion_coefficients']
            )

            image_width = self.camera_params[camera_name]['ros__parameters']['image_width']
            image_height = self.camera_params[camera_name]['ros__parameters']['image_height']
            
            # Get the new intrinsic matrix after fisheye distortion correction keeping same image size
            intrinsic_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                intrinsic_matrix, distortion_coeffs, (image_width, image_height), np.eye(3), balance=1.0)
            
            # Create frustum marker
            marker = Marker()
            marker.header.frame_id = camera_name
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            
            # Set scale
            marker.scale.x = 0.02  # Line width
            
            # Set color based on camera
            colors = {
                'frontCamera': [1.0, 0.0, 0.0, 1.0],  # Red
                'leftCamera': [0.0, 1.0, 0.0, 1.0],   # Green
                'rightCamera': [0.0, 0.0, 1.0, 1.0],  # Blue
                'rearCamera': [1.0, 1.0, 0.0, 1.0]    # Yellow
            }
            marker.color = ColorRGBA(
                r=colors[camera_name][0],
                g=colors[camera_name][1],
                b=colors[camera_name][2],
                a=colors[camera_name][3]
            )
            
            # Calculate frustum vertices
            frustum_points = self.calculate_frustum_vertices(
                intrinsic_matrix, image_width, image_height, max_distance=1.0
            )
            
            # Add frustum lines
            for point in frustum_points:
                marker.points.append(Point(x=point[0], y=point[1], z=point[2]))
            
            marker_array.markers.append(marker)
            
            # Add camera name text marker
            text_marker = Marker()
            text_marker.header.frame_id = camera_name
            text_marker.header.stamp = self.get_clock().now().to_msg()
            text_marker.id = i + 100
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = 0.0
            text_marker.pose.position.y = 0.0
            text_marker.pose.position.z = 0.1
            text_marker.scale.z = 0.2
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.text = camera_name
            marker_array.markers.append(text_marker)
        
        # Publish markers
        self.marker_publisher.publish(marker_array)
        self.get_logger().info("FOV markers published")
    
    def calculate_frustum_vertices(self, intrinsic_matrix, image_width, image_height, max_distance=5.0):
        """Calculate frustum vertices for camera FOV visualization"""
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]
        
        # Calculate field of view angles
        fov_x = 2 * math.atan(image_width / (2 * fx))
        fov_y = 2 * math.atan(image_height / (2 * fy))
        
        # Calculate frustum corners at max distance
        half_width = max_distance * math.tan(fov_x / 2)
        half_height = max_distance * math.tan(fov_y / 2)

        min_distance = 0.01
        
        # Define frustum vertices (camera coordinate system: X-rightward, Y-downward, Z-forward)
        vertices = [
            # Near plane (at distance 0.1m)
            [-min_distance * half_width / max_distance, -min_distance * half_height / max_distance, min_distance],
            [min_distance * half_width / max_distance, -min_distance * half_height / max_distance, min_distance],
            [min_distance * half_width / max_distance, min_distance * half_height / max_distance, min_distance],
            [-min_distance * half_width / max_distance, min_distance * half_height / max_distance, min_distance],
            
            # Far plane (at max_distance)
            [-half_width, -half_height, max_distance],
            [half_width, -half_height, max_distance],
            [half_width, half_height, max_distance],
            [-half_width, half_height, max_distance]
        ]
        
        # Define lines to connect the frustum
        lines = []
        
        # Near plane edges
        lines.extend([vertices[0], vertices[1]])
        lines.extend([vertices[1], vertices[2]])
        lines.extend([vertices[2], vertices[3]])
        lines.extend([vertices[3], vertices[0]])
        
        # Far plane edges
        lines.extend([vertices[4], vertices[5]])
        lines.extend([vertices[5], vertices[6]])
        lines.extend([vertices[6], vertices[7]])
        lines.extend([vertices[7], vertices[4]])
        
        # Connecting edges
        lines.extend([vertices[0], vertices[4]])
        lines.extend([vertices[1], vertices[5]])
        lines.extend([vertices[2], vertices[6]])
        lines.extend([vertices[3], vertices[7]])
        
        return lines
    
    def create_visualization_overlays(self):
        """Create visualization overlays showing calibration errors"""
        self.get_logger().info("Creating visualization overlays...")
        
        for camera_name in self.camera_names:
            self.get_logger().info(f"  Processing {camera_name}...")
            
            # Load camera image
            if camera_name not in self.image_files:
                self.get_logger().warn(f"  No image file specified for {camera_name}, skipping overlay")
                continue
            
            image_path = os.path.join(self.scripts_dir, self.image_files[camera_name])
            if not os.path.exists(image_path):
                self.get_logger().warn(f"  Image not found: {image_path}, skipping overlay")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                self.get_logger().warn(f"  Failed to load image: {image_path}, skipping overlay")
                continue
            
            overlay_image = image.copy()
            
            # Get camera parameters
            intrinsic_matrix = np.array(
                self.camera_params[camera_name]['ros__parameters']['intrinsic_matrix']
            ).reshape(3, 3)
            distortion_coeffs = np.array(
                self.camera_params[camera_name]['ros__parameters']['distortion_coefficients']
            )
            transform_matrix = np.array(
                self.camera_params[camera_name]['ros__parameters']['transformation_matrix']
            ).reshape(4, 4)
            
            # Extract rotation and translation
            rotation_matrix = transform_matrix[:3, :3]
            translation = transform_matrix[:3, 3]
            
            # Convert to OpenCV format (camera to world)
            R_cam_to_world = rotation_matrix.T
            t_cam_to_world = -rotation_matrix.T @ translation
            
            # Convert distortion coefficients to fisheye format
            fisheye_dist_coeffs = self.convert_to_fisheye_distortion(distortion_coeffs)
            
            total_error = 0
            point_count = 0
            
            # Determine which point sets this camera uses
            if camera_name == 'frontCamera':
                point_sets = ['FL', 'FR']
            elif camera_name == 'leftCamera':
                point_sets = ['FL', 'RL']
            elif camera_name == 'rightCamera':
                point_sets = ['FR', 'RR']
            elif camera_name == 'rearCamera':
                point_sets = ['RL', 'RR']
            
            # Process each point set for this camera
            for point_set in point_sets:
                world_pts = self.world_points[point_set]
                expected_img_pts = self.image_points[camera_name][point_set]
                
                # Project world points to image using fisheye camera model
                world_pts_formatted = world_pts.astype(np.float32).reshape(-1, 1, 3)
                intrinsic_matrix_f32 = intrinsic_matrix.astype(np.float32)
                fisheye_dist_coeffs_f32 = fisheye_dist_coeffs.astype(np.float32)
                
                projected_points, _ = cv2.fisheye.projectPoints(
                    world_pts_formatted,
                    cv2.Rodrigues(R_cam_to_world)[0],
                    t_cam_to_world,
                    intrinsic_matrix_f32,
                    fisheye_dist_coeffs_f32
                )
                
                projected_points = projected_points.reshape(-1, 2)
                
                # Draw points and error vectors
                for i, (expected, projected) in enumerate(zip(expected_img_pts, projected_points)):
                    # Check if points are within image bounds
                    if (0 <= projected[0] < overlay_image.shape[1] and
                        0 <= projected[1] < overlay_image.shape[0]):
                        
                        # Draw expected point (green circle)
                        cv2.circle(overlay_image, tuple(expected.astype(int)), 8, (0, 255, 0), 2)
                        cv2.putText(
                            overlay_image, f"E{i+1}",
                            (int(expected[0]) + 10, int(expected[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                        )
                        
                        # Draw projected point (red circle)
                        cv2.circle(overlay_image, tuple(projected.astype(int)), 8, (0, 0, 255), 2)
                        cv2.putText(
                            overlay_image, f"P{i+1}",
                            (int(projected[0]) + 10, int(projected[1]) + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                        )
                        
                        # Draw error vector
                        cv2.arrowedLine(
                            overlay_image,
                            tuple(expected.astype(int)),
                            tuple(projected.astype(int)),
                            (255, 0, 255), 2, tipLength=0.3
                        )
                        
                        # Calculate and display error
                        error = np.linalg.norm(projected - expected)
                        cv2.putText(
                            overlay_image, f"{error:.1f}px",
                            (int((expected[0] + projected[0])/2), int((expected[1] + projected[1])/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                        )
                        
                        self.get_logger().info(
                            f"    {point_set} Point {i+1}: Expected {expected}, "
                            f"Projected {projected}, Error: {error:.2f} pixels"
                        )
                        total_error += error
                        point_count += 1
                    else:
                        self.get_logger().warn(
                            f"    {point_set} Point {i+1}: Projected point {projected} is outside image bounds"
                        )
            
            avg_error = total_error / point_count if point_count > 0 else 0
            self.get_logger().info(f"  {camera_name} average reprojection error: {avg_error:.2f} pixels")
            
            # Add legend
            legend_y = 30
            cv2.putText(
                overlay_image, "Green: Expected points", (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                overlay_image, "Red: Projected points", (10, legend_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            cv2.putText(
                overlay_image, "Magenta: Error vectors", (10, legend_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
            )
            cv2.putText(
                overlay_image, f"Avg Error: {avg_error:.1f}px", (10, legend_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            # Save overlay image
            output_path = os.path.join(self.scripts_dir, f"{camera_name}_calibration_debug.jpg")
            cv2.imwrite(output_path, overlay_image)
            self.get_logger().info(f"  Saved debug overlay: {output_path}")
    
    def publish_loop(self):
        """Publish markers periodically for visualization"""
        while self.running and rclpy.ok():
            self.create_fov_markers()
            time.sleep(1.0)  # Publish every second
  
    def convert_to_fisheye_distortion(self, distortion_coeffs):
        """Convert distortion coefficients to fisheye format"""
        if len(distortion_coeffs) >= 4:
            fisheye_coeffs = np.array(
                [distortion_coeffs[0], distortion_coeffs[1],
                 distortion_coeffs[2], distortion_coeffs[3]],
                dtype=np.float32
            )
        else:
            # Pad with zeros if we have fewer coefficients
            fisheye_coeffs = np.zeros(4, dtype=np.float32)
            fisheye_coeffs[:len(distortion_coeffs)] = distortion_coeffs
        
        return fisheye_coeffs
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        trace = np.trace(R)
        
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        
        return np.array([qx, qy, qz, qw])


def main(args=None):
    rclpy.init(args=args)
    
    node = UnifiedCameraCalibrationNode()
    
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     node.running = False
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()


    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

