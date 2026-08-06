#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <functional>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <optional>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

class PointCloudFilterNode : public rclcpp::Node{
    public:
        PointCloudFilterNode() : rclcpp::Node("pointcloud_filter_node"){
            //Voxel downsampling
            this->declare_parameter("voxel_leaf", 0.03);
            //2D grid and local ground estimation
            this->declare_parameter("grid_res", 0.15);
            this->declare_parameter("neighbor_radius", 1);
            //Basic point classification
            this->declare_parameter("height_thresh", 0.14); //0.16
            this->declare_parameter("max_obstacle_z", 2.0);
            this->declare_parameter("min_obstacle_height", 0.0);
            this->declare_parameter("strong_obstacle_height", 0.50);
            //Obstacle-cell mask
            this->declare_parameter("vertical_range_thresh", 0.4);
            this->declare_parameter("ground_jump_thresh", 0.18);
            this->declare_parameter("obstacle_cell_inflation", 1);
            //Ground points inside cells that also contain obstacles
            this->declare_parameter("exclude_obstacle_cells_from_ground", true);
            this->declare_parameter("obstacle_cell_ground_keep_height", 0.06);
            //Local plane refinement - currently not used
            /*
            this->declare_parameter("enable_plane_refinement", true);
            this->declare_parameter("plane_refinement_radius", 2);
            this->declare_parameter("plane_refinement_min_points", 5); 
            this->declare_parameter("plane_refinement_max_residual", 0.08);
            this->declare_parameter("plane_refinement_max_slope", 0.45);
            this->declare_parameter("plane_refine_ground_like_obstacle_cells", true);            
            */
            //Spatial filter for obstacle candidates
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

            bool operator==(const CellIndex& other) const noexcept{
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
                
                auto it = height_cells.find(key);
                if(it == height_cells.end()){
                    height_cells.emplace(key, CellStats{point.z, point.z});
                    continue;
                }
                if(it->second.min_z >= point.z){
                    it->second.min_z = point.z;
                }
                else if(it->second.max_z <= point.z){
                    it->second.max_z = point.z;
                }
            }
            
            //Obstacle cell mask
            //Mask containing all cell which meet requirements:
            //vertical_range > vertical_range_thresh && ground_jump < ground_jump_thresh
            std::unordered_set<CellIndex, CellIndexHash> obstacle_cell_mask;
            obstacle_cell_mask.reserve(height_cells.size());

            for(const auto& [index, stat] : height_cells){
                float vertical_range = stat.max_z - stat.min_z;
                float neighbor_median_minz = findNeighborMedianMinZ(height_cells, index);
                float ground_jump = std::abs(stat.min_z - neighbor_median_minz);

                if(vertical_range > vertical_range_thresh && ground_jump < ground_jump_thresh){
                    //optinal inflation arround marked cell
                    for(int dx = -obstacle_cell_inflation; dx <= obstacle_cell_inflation; ++dx){
                        for(int dy = -obstacle_cell_inflation; dy <= obstacle_cell_inflation; ++dy){

                            CellIndex neighbor{index.x + dx, index.y + dy};
                            auto it = height_cells.find(neighbor);
                            if(it == height_cells.end()) continue;

                            obstacle_cell_mask.insert(neighbor);
                        }
                    }
                }
            }

            //first step of classification (individual points)
            pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points(new pcl::PointCloud<pcl::PointXYZ>);
            ground_points->points.reserve(cloud_downsampled->points.size());

            std::unordered_map<CellIndex, std::vector<pcl::PointXYZ>, CellIndexHash> candidate_obstacle_points_by_cell;
            candidate_obstacle_points_by_cell.reserve(height_cells.size());

            std::unordered_set<CellIndex, CellIndexHash> current_obstacle_cells;
             
            for(const auto& point : cloud_downsampled->points){
                if(!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;

                CellIndex index{
                    static_cast<int>(std::floor(point.x /grid_res)),
                    static_cast<int>(std::floor(point.y /grid_res))
                };

                std::optional<float> local_ground_z = findLocalGroundZ(height_cells, index);

                if(!local_ground_z) continue;

                const float dz = point.z - *local_ground_z;

                bool in_obstacle_cell = (obstacle_cell_mask.find(index) != obstacle_cell_mask.end());
                //hardcoded negative tolerance of 0.03 (3cm)
                //e.g. point.z can be cause of sensor noise 
                bool is_ground_like = (-0.03f <= dz) && (dz < height_thresh);

                //aditional condition for dealing with 2 extreme cases:
                //keeping all ground-like points: obstacles can be in ground cloud
                //discarding all ground-like points in obstacle mask: losing ground points near obstacle
                const bool keep_ground_in_obstacle_cell =
                    !exclude_obstacle_cells_from_ground || 
                    !in_obstacle_cell || 
                    dz <= obstacle_cell_ground_keep_height;

                bool is_obstacle_candidate = false;

                if(is_ground_like){
                    if(keep_ground_in_obstacle_cell){
                        ground_points->points.push_back(point);
                    }
                }
                else if(point.z < max_obstacle_z &&
                    dz >= min_obstacle_height && 
                    (in_obstacle_cell || dz >= strong_obstacle_height)){
                    is_obstacle_candidate = true;
                }
                
                if(is_obstacle_candidate){
                    current_obstacle_cells.insert(index);
                    candidate_obstacle_points_by_cell[index].push_back(point);
                }
            }

            //Spatial filter
            std::unordered_set<CellIndex, CellIndexHash> filtered_obstacle_cells;
            for(CellIndex index : current_obstacle_cells){
                int neighbor_count = countObstacleNeighbors(current_obstacle_cells, index);
                if (neighbor_count >= min_obstacle_neighbor_cells) filtered_obstacle_cells.insert(index);
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_points(new pcl::PointCloud<pcl::PointXYZ>);
            obstacle_points->points.reserve(cloud_downsampled->points.size());
            for(const auto& [index, points]: candidate_obstacle_points_by_cell){
                auto it = filtered_obstacle_cells.find(index);
                if (it != filtered_obstacle_cells.end()){
                    for(const auto& point : points){
                        obstacle_points->points.push_back(point);
                    }
                }
            }

            //PCL metadata
            ground_points->width = ground_points->points.size();
            ground_points->height = 1;
            ground_points->is_dense = true;

            obstacle_points->width = obstacle_points->points.size();
            obstacle_points->height = 1;
            obstacle_points->is_dense = true;

            //Publish
            sensor_msgs::msg::PointCloud2 ground_msg;
            sensor_msgs::msg::PointCloud2 obstacle_msg;

            pcl::toROSMsg(*ground_points, ground_msg);
            pcl::toROSMsg(*obstacle_points, obstacle_msg);

            ground_msg.header = msg->header;
            obstacle_msg.header = msg->header;

            ground_pub_->publish(ground_msg);
            obstacle_pub_->publish(obstacle_msg);
            
        }

        //Find Median of min_z of cells in given radius around given cell
        float findNeighborMedianMinZ(const std::unordered_map<CellIndex, CellStats, CellIndexHash>& height_cells, const CellIndex& index){
            int radius = 1; //add param later

            std::vector<float> neighbors;
            neighbors.reserve((2 * radius + 1) * (2 * radius + 1) -1);

            for(int dx = -radius; dx <= radius; ++dx){
                for(int dy = -radius; dy <= radius; ++dy){
                    if (dx == 0 && dy == 0) continue;
                    
                    CellIndex neighbor{index.x + dx, index.y + dy};
                    auto it = height_cells.find(neighbor);
                    if(it == height_cells.end()) continue;

                    neighbors.push_back(it->second.min_z);
                }
            }

            if(neighbors.empty()) return height_cells.at(index).min_z;

            const std::size_t median_index = neighbors.size() / 2;
            std::nth_element(neighbors.begin(), neighbors.begin() + median_index, neighbors.end());

            return neighbors[median_index];
            
        }

        //Find min min_z in given radius around given cell
        std::optional<float> findLocalGroundZ(const std::unordered_map<CellIndex, CellStats, CellIndexHash>& height_cells, const CellIndex& index){
            int radius = 1; //add param later
            std::optional<float> min_z;

            for(int dx = -radius; dx <= radius; ++dx){
                for(int dy = -radius; dy <= radius; ++dy){
                    CellIndex neighbor{index.x + dx, index.y +dy};
                    auto it = height_cells.find(neighbor);
                    if(it == height_cells.end()) continue;

                    if(!min_z || it->second.min_z < *min_z) min_z =  it->second.min_z;
                }
            }
            return min_z;

        }

        //Count how many neighbors of obstacle cell are also marked as obstacle
        int countObstacleNeighbors(const std::unordered_set<CellIndex, CellIndexHash>& obstacle_cells, const CellIndex& index){
            int radius = 1; //add param later
            int count = 0;

            for(int dx = -radius; dx <= radius; ++dx){
                for(int dy = -radius; dy <= radius; ++dy){
                    if(dx == 0 && dy == 0) continue;
                    CellIndex neighbor{index.x +dx, index.y +dy};
                    auto it = obstacle_cells.find(neighbor);
                    if(it != obstacle_cells.end()) count++; 
                }
            }

            return count;
        }

        
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudFilterNode>());
    rclcpp::shutdown();
    return 0;
}
