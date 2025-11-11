#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='surround_vision',
            executable='fast_image_republisher',
            name='fast_image_republisher',
            output='screen',
            parameters=[{
                'camera_names': ['front', 'left', 'rear', 'right', 'inside', 'ground'],
                'use_gpu': False,
                'use_libjpeg_turbo': True,
                'enable_timing': True,
                'reverse_color_channels': False,
                'qos_depth': 1,
                'max_image_width': 1920,
                'max_image_height': 1080,
            }],
        )
    ])
