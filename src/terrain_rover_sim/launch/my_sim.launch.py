import os

from ament_index_python.packages import get_package_share_directory

from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from launch_ros.actions import Node

import xacro


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    pkg_terrain_rover_sim = get_package_share_directory("terrain_rover_sim")

    default_world = os.path.join(pkg_terrain_rover_sim, "worlds", "my_world.sdf")
    gazebo_resource_path = os.path.dirname(pkg_terrain_rover_sim)

    sim_world_arg = DeclareLaunchArgument(
        "sim_world",
        default_value=default_world,
        description="Path to custom Gazebo world file",
    )

    enable_ground_truth_arg = DeclareLaunchArgument(
        "enable_ground_truth",
        default_value="true",
        description="Publish Gazebo ground-truth odometry and map->odom TF",
    )

    gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[
            gazebo_resource_path,
            ":",
            EnvironmentVariable("GZ_SIM_RESOURCE_PATH", default_value=""),
        ],
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-r ", LaunchConfiguration("sim_world")]
        }.items(),
    )
   
    robot_desc = xacro.process(
        os.path.join(
            pkg_terrain_rover_sim,
            "urdf",
            "rover_sim.urdf.xacro",
        ),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[
            {"use_sim_time": True},
            {"robot_description": robot_desc},
        ],
    )
    
    robot_gazebo_name = "rover"
    
    spawn_rover = Node(
        package="ros_gz_sim",
        executable="create",
        name="ros_gz_sim_create",
        output="both",
        arguments=[
            "-topic", "robot_description",
            "-name", robot_gazebo_name,
            "-x", "0",
            "-y", "0",
            "-z", "1.65",
        ],
    )

    topic_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="parameter_bridge",
        arguments=[
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/imu/data_raw@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            "/rgbd_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        output="screen",
    )

    image_bridge = Node(
        package="ros_gz_image",
        executable="image_bridge",
        name="image_bridge",
        arguments=[
            "/rgbd_camera/image",
            "/rgbd_camera/depth_image",
        ],
        output="screen",
    )

    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg_terrain_rover_sim, 'config', 'ekf.yaml'),
            {'use_sim_time': True},
            {'publish_tf': False},
        ]
    )

    raw_ground_truth_topic = "/ground_truth/raw_tf"
    ground_truth_odom_topic = "/ground_truth/odom"
    odom_source_topic = "/odom"
    base_frame = "base_footprint"

    ground_truth_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ground_truth_bridge",
        arguments=[
            "/world/terrain_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
        ],
        remappings=[
            ("/world/terrain_world/dynamic_pose/info", raw_ground_truth_topic),
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_ground_truth")),
    )

    ground_truth_node = Node(
        package="terrain_rover_sim",
        executable="ground_truth_node",
        name="ground_truth_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"input_topic": raw_ground_truth_topic},
            {"odom_topic": ground_truth_odom_topic},
            {"odom_source_topic": odom_source_topic},
            {"model_frame": robot_gazebo_name},
            {"map_frame": "map"},
            {"odom_frame": "odom"},
            {"base_frame": base_frame},
            {"publish_odom_to_base_tf": True},
            {"publish_map_to_odom_tf": True},
        ],
        condition=IfCondition(LaunchConfiguration("enable_ground_truth")),
    )

    return LaunchDescription(
        [
            sim_world_arg,
            enable_ground_truth_arg,
            gz_resource_path,
            gz_sim,
            robot_state_publisher,
            spawn_rover,
            topic_bridge,
            image_bridge,
            robot_localization_node,
            ground_truth_bridge,
            ground_truth_node,
        ]
    )
