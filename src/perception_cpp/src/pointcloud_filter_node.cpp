#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <functional>

class PointCloudFilterNode : public rclcpp::Node{
    public:
        PointCloudFilterNode() : rclcpp::Node("pointcloud_filter_node"){
            // Voxel downsampling
            this->declare_parameter("voxel_leaf", 0.05);
            // 2D grid and local ground estimation
            this->declare_parameter("grid_res", 0.15);
            this->declare_parameter("neighbor_radius", 1);
            // Basic point classification
            this->declare_parameter("height_thresh", 0.16);
            this->declare_parameter("max_obstacle_z", 2.0);
            this->declare_parameter("min_obstacle_height", 0.30);
            this->declare_parameter("strong_obstacle_height", 0.50);
            // Obstacle-cell mask
            this->declare_parameter("vertical_range_thresh", 0.4);
            this->declare_parameter("ground_jump_thresh", 0.18);
            this->declare_parameter("obstacle_cell_inflation", 1);
            // Ground points inside cells that also contain obstacles
            this->declare_parameter("exclude_obstacle_cells_from_ground", true);
            this->declare_parameter("obstacle_cell_ground_keep_height", 0.06);
            // Local plane refinement
            this->declare_parameter("enable_plane_refinement", true);
            this->declare_parameter("plane_refinement_radius", 2);
            this->declare_parameter("plane_refinement_min_points", 5);
            this->declare_parameter("plane_refinement_max_residual", 0.08);
            this->declare_parameter("plane_refinement_max_slope", 0.45);
            this->declare_parameter("plane_refine_ground_like_obstacle_cells", true);
            // Spatial filter for obstacle candidates.
            this->declare_parameter("obstacle_neighbor_radius", 1);
            this->declare_parameter("min_obstacle_neighbor_cells", 1);

            //subscription
            raw_cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/scan/points",
                10,
                [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){this->pointCloudCallback(msg);}
                );
            //publishers 
            ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points_ground",10);
            obstacle_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/points_obstacles",10);
            
        }
    private:
        struct CellStats{
            float min_z;
            float max_z;
        };
        struct CellIndex{
            int x;
            int y;

            bool operator==(const CellIndex other) const{
                return x == other.x && y == other.y;
            }
        };
        struct CellIndexHash {
            std::size_t operator()(const CellIndex& cell) const {
                const std::size_t h1 = std::hash<int>{}(cell.x);
                const std::size_t h2 = std::hash<int>{}(cell.y);

                return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            }
        };
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr raw_cloud_sub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;

        void pointCloudCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){
            
            double grid_res = this->get_parameter("grid_res").as_double();
            double height_thresh = this->get_parameter("height_thresh").as_double();
            int neighbor_radius = this->get_parameter("neighbor_radius").as_int();
            double vertical_range_thresh = this->get_parameter("vertical_range_thresh").as_double();
            double ground_jump_thresh = this->get_parameter("ground_jump_thresh").as_double();
            int obstacle_cell_inflation = this->get_parameter("obstacle_cell_inflation").as_int();
            double voxel_leaf = this->get_parameter("voxel_leaf").as_double();
            double max_obstacle_z = this->get_parameter("max_obstacle_z").as_double();
            double min_obstacle_height = this->get_parameter("min_obstacle_height").as_double();
            double strong_obstacle_height = this->get_parameter("strong_obstacle_height").as_double();
            bool exclude_obstacle_cells_from_ground = this->get_parameter("exclude_obstacle_cells_from_ground").as_bool();
            double obstacle_cell_ground_keep_height = this->get_parameter("obstacle_cell_ground_keep_height").as_double();
            bool enable_plane_refinement = this->get_parameter("enable_plane_refinement").as_bool();
            int plane_refinement_radius = this->get_parameter("plane_refinement_radius").as_int();
            int plane_refinement_min_points = this->get_parameter("plane_refinement_min_points").as_int();
            double plane_refinement_max_residual = this->get_parameter("plane_refinement_max_residual").as_double();
            double plane_refinement_max_slope = this->get_parameter("plane_refinement_max_slope").as_double();
            bool plane_refine_ground_like_obstacle_cells = this->get_parameter("plane_refine_ground_like_obstacle_cells").as_bool();
            int obstacle_neighbor_radius = this->get_parameter("obstacle_neighbor_radius").as_int();
            int min_obstacle_neighbor_cells = this->get_parameter("min_obstacle_neighbor_cells").as_int();

            //Voxel Grid
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud_input);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
            //Can be used to turn off voxel downsampling with parameters
            if(voxel_leaf <= 0){
                cloud_downsampled = cloud_input;
            }
            else{
                pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
                voxel_filter.setInputCloud(cloud_input);
                voxel_filter.setLeafSize(voxel_leaf,voxel_leaf,voxel_leaf);
                voxel_filter.filter(*cloud_downsampled);
            }

            //2D Grid containing min and max z
            if(grid_res <= 0){
                RCLCPP_DEBUG(this->get_logger(), "Invalid grid resolution!");
                return;
            }

            std::unordered_map<CellIndex, CellStats, CellIndexHash> height_cells;
            height_cells.reserve(cloud_downsampled->points.size());

            for(const auto& point : cloud_downsampled->points){
                if(!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;

                CellIndex key;
                key.x = static_cast<int>(std::floor(point.x / grid_res));
                key.y = static_cast<int>(std::floor(point.y / grid_res));

                if(!height_cells.count(key)){
                    height_cells[key].min_z = point.z;
                    height_cells[key].max_z = point.z;
                    continue;
                }
                if(height_cells[key].min_z >= point.z){
                    height_cells[key].min_z = point.z;
                }
                else if(height_cells[key].max_z <= point.z){
                    height_cells[key].max_z = point.z;
                }
            }

            
        }

        
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudFilterNode>());
    rclcpp::shutdown();
    return 0;
}
