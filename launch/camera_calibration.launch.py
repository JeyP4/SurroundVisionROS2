#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("surround_vision")
    rviz_config = os.path.join(package_share, "rviz", "calibration_debug.rviz")
    
    # Get the path to the unified calibration script
    calibration_script = os.path.join(package_share, "camera_calibration", "unified_camera_calibration.py")
    
    # Verify the script exists
    if not os.path.exists(calibration_script):
        raise FileNotFoundError(
            f"Calibration script not found at: {calibration_script}\n"
            f"Please ensure the script exists at the expected location."
        )
    
    # Get URDF file path and read it
    urdf_file = os.path.join(package_share, "urdf", "vehicle.urdf")
    
    # Read URDF file content
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()
    
    # Robot State Publisher to publish the vehicle model
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': False,
            'robot_description': robot_desc
        }]
    )
    
    # Launch Rviz
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )
    
    # Launch the calibration script
    # Use ExecuteProcess to run the Python script
    calibration_process = ExecuteProcess(
        cmd=['python3', calibration_script],
        output='screen',
        name='unified_camera_calibration'
    )
    
    # Delay the calibration script slightly to ensure Rviz is ready
    # This gives Rviz time to start and connect to ROS2
    delayed_calibration = TimerAction(
        period=2.0,  # Wait 2 seconds for Rviz to start
        actions=[calibration_process]
    )
    
    return LaunchDescription([
        robot_state_publisher,
        rviz_node,
        delayed_calibration
    ])

