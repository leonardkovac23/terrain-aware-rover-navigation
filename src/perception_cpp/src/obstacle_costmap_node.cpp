#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "std_msgs/msg/header.hpp"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"

class ObstacleCostmapNode : public rclcpp::Node{
    public:
        ObstacleCostmapNode() : rclcpp::Node("obstacle_costmap_node"){
            //ROS interface and frame setup
            this->declare_parameter("input_topic", "/points_obstacles");
            this->declare_parameter("map_topic", "/map/obstacles");
            this->declare_parameter("target_frame", "map");
            //Fixed 2D map geometry in target_frame
            this->declare_parameter("grid_resolution", 0.35);
            this->declare_parameter("map_width", 120.0);
            this->declare_parameter("map_height", 120.0);
            this->declare_parameter("origin_x", -60.0);
            this->declare_parameter("origin_y", -60.0);
            //Point insertion filters before updating obstacle hits
            this->declare_parameter("min_insert_range", 0.20);
            this->declare_parameter("max_insert_range", 8.0);
            this->declare_parameter("min_target_z", -10.0);
            this->declare_parameter("max_target_z", 10.0);
            //Temporal hit accumulation for persistent obstacle cells
            this->declare_parameter("hit_increment", 1);
            this->declare_parameter("min_hits_for_obstacle", 2);
            this->declare_parameter("max_hits_per_cell", 50);
            this->declare_parameter("enable_hit_decay", false);
            this->declare_parameter("hit_decay_per_publish", 0);
            //Per-scan spatial validation of obstacle cells
            this->declare_parameter("min_points_per_cell", 2);
            this->declare_parameter("obstacle_neighbor_radius", 1);
            this->declare_parameter("min_obstacle_neighbor_cells", 1);
            //OccupancyGrid output values and obstacle inflation
            this->declare_parameter("obstacle_cost", 100);
            this->declare_parameter("free_cost", 0);
            this->declare_parameter("inflation_radius", 0.25);
            //Runtime behavior
            this->declare_parameter("publish_rate", 2.0);
            this->declare_parameter("tf_lookup_timeout", 0.30);

            const std::string input_topic = this->get_parameter("input_topic").as_string();
            const std::string map_topic = this->get_parameter("map_topic").as_string();

            //subscription
            obstacle_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                input_topic,
                10,
                [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){this->obstacleCostmapCallback(msg);}
            );
            //publisher
            map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic,10);
        }
    private:
        struct CellIndex{
            int x;
            int y;

            bool operator==(const CellIndex& other) const noexcept{
                return x == other.x && y == other.y;
            }
        };
        struct CellIndexHash{
            std::size_t operator()(const CellIndex& cell) const {
                const std::size_t h1 = std::hash<int>{}(cell.x);
                const std::size_t h2 = std::hash<int>{}(cell.y);

                return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
            }
        };
        
        std::unordered_map<CellIndex, int, CellIndexHash> hit_counts_;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_points_sub_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

        void obstacleCostmapCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){
            
            std::string target_frame = this->get_parameter("target_frame").as_string();
            double grid_resolution = this->get_parameter("grid_resolution").as_double();
            double map_width = this->get_parameter("map_width").as_double();
            double map_height = this->get_parameter("map_height").as_double();
            double origin_x = this->get_parameter("origin_x").as_double();
            double origin_y = this->get_parameter("origin_y").as_double();
            double min_insert_range = this->get_parameter("min_insert_range").as_double();
            double max_insert_range = this->get_parameter("max_insert_range").as_double();
            double min_target_z = this->get_parameter("min_target_z").as_double();
            double max_target_z = this->get_parameter("max_target_z").as_double();
            int hit_increment = this->get_parameter("hit_increment").as_int();
            int min_hits_for_obstacle = this->get_parameter("min_hits_for_obstacle").as_int();
            int max_hits_per_cell = this->get_parameter("max_hits_per_cell").as_int();
            bool enable_hit_decay = this->get_parameter("enable_hit_decay").as_bool();
            int hit_decay_per_publish = this->get_parameter("hit_decay_per_publish").as_int();
            int min_points_per_cell = this->get_parameter("min_points_per_cell").as_int();
            int obstacle_neighbor_radius = this->get_parameter("obstacle_neighbor_radius").as_int();
            int min_obstacle_neighbor_cells = this->get_parameter("min_obstacle_neighbor_cells").as_int();
            int obstacle_cost = this->get_parameter("obstacle_cost").as_int();
            int free_cost = this->get_parameter("free_cost").as_int();
            double inflation_radius = this->get_parameter("inflation_radius").as_double();
            double publish_rate = this->get_parameter("publish_rate").as_double();
            double tf_lookup_timeout = this->get_parameter("tf_lookup_timeout").as_double();

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud_input);

            std::unordered_map<CellIndex, int, CellIndexHash> point_counts_by_cell;

            if(grid_resolution <= 0.0) {
                RCLCPP_WARN(this->get_logger(), "Invalid grid_resolution");
                return;
            }

            int height_cells = static_cast<int>(std::round(map_height / grid_resolution));
            int width_cells = static_cast<int>(std::round(map_width / grid_resolution));
            const double min_range_sq = min_insert_range*min_insert_range;
            const double max_range_sq = max_insert_range*max_insert_range;

            for(const auto& point : cloud_input->points){
                if(!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;

                if(point.z < min_target_z || point.z > max_target_z) continue;
                
                const double range_sq = point.x*point.x + point.y*point.y + point.z*point.z;

                if(range_sq < min_range_sq || range_sq > max_range_sq) continue;


                int ix = static_cast<int>(std::floor((point.x - origin_x) / grid_resolution));
                int iy = static_cast<int>(std::floor((point.y - origin_y) / grid_resolution));
                //Out of bounds check
                if(ix < 0 || ix >= width_cells || iy <0 || iy >= height_cells) continue;

                CellIndex index{ix, iy};
                point_counts_by_cell[index]++;
            }

            std::unordered_set<CellIndex, CellIndexHash> obstacle_candidates;
            for(const auto& [index, count] : point_counts_by_cell){
                if(count >= min_points_per_cell) obstacle_candidates.insert(index);
            }

            for(const auto& index : obstacle_candidates){
                if(countCandidateNeighbors(obstacle_candidates, index, obstacle_neighbor_radius) >= min_obstacle_neighbor_cells){
                    hit_counts_[index] += hit_increment;

                    if(hit_counts_[index] > max_hits_per_cell){
                        hit_counts_[index] = max_hits_per_cell;
                    }
                }
            }
        }

        //Count how many neighbors of obstacle cell are also marked as obstacle
        int countCandidateNeighbors(const std::unordered_set<CellIndex, CellIndexHash>& obstacle_candidates, const CellIndex& index, int radius){
            int count = 0;

            for(int dx = -radius; dx <= radius; ++dx){
                for(int dy = -radius; dy <= radius; ++dy){
                    if(dx == 0 && dy == 0) continue;
                    CellIndex neighbor{index.x +dx, index.y +dy};
                    auto it = obstacle_candidates.find(neighbor);
                    if(it != obstacle_candidates.end()) count++; 
                }
            }

            return count;
        }
};
