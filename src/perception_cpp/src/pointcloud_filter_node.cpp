#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

class PointCloudFilterNode : public rclcpp::Node{
    public:
        PointCloudFilterNode() : rclcpp::Node("pointcloud_filter_node"){
            this->declare_parameter("grid_res", 0.15);
            this->declare_parameter("height_thresh", 0.16);
            this->declare_parameter("neighbor_radius", 1);
            this->declare_parameter("vertical_range_thresh", 0.4);
            this->declare_parameter("ground_jump_thresh", 0.18);
            this->declare_parameter("obstacle_cell_inflation", 1);
            this->declare_parameter("voxel_leaf", 0.05);
            this->declare_parameter("max_obstacle_z", 2.0);
            this->declare_parameter("min_obstacle_height", 0.30);
            this->declare_parameter("strong_obstacle_height", 0.50);
            this->declare_parameter("exclude_obstacle_cells_from_ground", true);
            this->declare_parameter("obstacle_cell_ground_keep_height", 0.06);
            this->declare_parameter("enable_plane_refinement", true);
            this->declare_parameter("plane_refinement_radius", 2);
            this->declare_parameter("plane_refinement_min_points", 5);
            this->declare_parameter("plane_refinement_max_residual", 0.08);
            this->declare_parameter("plane_refinement_max_slope", 0.45);
            this->declare_parameter("plane_refine_ground_like_obstacle_cells", true);
            this->declare_parameter("obstacle_neighbor_radius", 1);
            this->declare_parameter("min_obstacle_neighbor_cells", 1);

            raw_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/scan/points",
                10,
                [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){this->pointCloudCallback(msg);}
                );
            
            ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points_ground",10);
            obstacle_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points_obstacles",10);
            

        }
    private:
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr raw_cloud_sub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;

        void pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){
            
        }
        
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudFilterNode>());
    rclcpp::shutdown();
    return 0;
}