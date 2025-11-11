#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    surround_vision_dir = get_package_share_directory("surround_vision")
    default_config = os.path.join(surround_vision_dir, "config", "surround_vision_compressed.yaml")
    # default_config = os.path.join(surround_vision_dir, "config", "surround_vision_raw.yaml")

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=TextSubstitution(text=default_config),
        description="Path to a YAML file containing parameters for surround_vision_node",
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock if true",
    )

    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="Logging level passed to surround_vision_node",
    )

    node = Node(
        package="surround_vision",
        executable="surround_vision_node",
        name="surround_vision_node",
        output="screen",
        parameters=[
            LaunchConfiguration("config"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    return LaunchDescription([config_arg, use_sim_time_arg, log_level_arg, node])