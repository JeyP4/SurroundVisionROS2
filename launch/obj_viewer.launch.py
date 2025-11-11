import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory("surround_vision")
    default_config = os.path.join(package_share, "config", "obj_viewer.yaml")

    obj_viewer = Node(
        package="surround_vision",
        executable="obj_viewer_node",
        name="obj_viewer_node",
        output="screen",
        parameters=[default_config],
    )

    return LaunchDescription([obj_viewer])

