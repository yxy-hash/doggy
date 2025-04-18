import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("dog"),
        'config',
        'dog_cfg.yaml'
    )
    
    return LaunchDescription(
        [
            Node(
                package="inference",
                executable="inference_node",
                name="infer",
                namespace="dog",
                parameters=[
                    config
                ]
            )
            
            
        ]
    )