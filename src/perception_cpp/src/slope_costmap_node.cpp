#include <chrono>
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "std_msgs/msg/header.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl_conversions/pcl_conversions.h"

#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/exceptions.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

class SlopeCostmapNode : public rclcpp::Node{
    public:
        SlopeCostmapNode() : rclcpp::Node("slope_costmap_node"){
            //ROS interface and frame setup
            this->declare_parameter("input_topic", "/points_ground");
            this->declare_parameter("map_topic", "/map/slope");
            this->declare_parameter("target_frame", "map");
            //Fixed 2D map geometry in target_frame
            this->declare_parameter("grid_resolution", 0.35);
            this->declare_parameter("map_width", 120.0);
            this->declare_parameter("map_height", 120.0);
            this->declare_parameter("origin_x", -60.0);
            this->declare_parameter("origin_y", -60.0);
            //Source cloud range filtering before height map insertion
            this->declare_parameter("min_insert_range", 0.60);
            this->declare_parameter("max_insert_range", 8.0);
            //Persistent height map update parameters
            this->declare_parameter("height_alpha", 0.20);
            this->declare_parameter("max_height_update_jump", 0.35);
            this->declare_parameter("outlier_accept_count", 4);
            this->declare_parameter("outlier_same_height_tolerance", 0.15);
            this->declare_parameter("outlier_alpha", 0.20);
            this->declare_parameter("min_cell_confidence", 2);
            this->declare_parameter("max_cell_confidence", 20);
            this->declare_parameter("max_batch_z_std", 0.20);
            this->declare_parameter("max_batch_z_range", 0.45);
            this->declare_parameter("height_percentile", 30.0);
            //Slope calculation
            this->declare_parameter("slope_radius", 2);
            this->declare_parameter("min_neighbors_for_slope", 3);
            this->declare_parameter("min_neighbor_fill_ratio", 0.15);
            this->declare_parameter("max_neighbor_height_range", 1.50);
            this->declare_parameter("max_neighbor_height_std", 0.45);
            this->declare_parameter("slope_safe", 0.087);
            this->declare_parameter("slope_caution", 0.174);
            this->declare_parameter("slope_danger", 0.262);
            //OccupancyGrid output cost values
            this->declare_parameter("cost_safe", 0);
            this->declare_parameter("cost_caution", 50);
            this->declare_parameter("cost_danger", 80);
            this->declare_parameter("cost_lethal", 95);
            this->declare_parameter("cost_unknown", -1);
            //Cost filtering and persistence
            this->declare_parameter("cost_smoothing_alpha_up", 1.0);
            this->declare_parameter("cost_smoothing_alpha_down", 1.0);
            this->declare_parameter("enable_cost_decay", false);
            this->declare_parameter("cost_decay_per_publish", 0);
            this->declare_parameter("enable_hole_filling", false);
            this->declare_parameter("hole_fill_min_known_neighbors", 6);
            //Warmup behavior before publishing stable maps
            this->declare_parameter("warmup_clouds", 10);
            this->declare_parameter("enable_publish_during_warmup", false);
            //Runtime behavior
            this->declare_parameter("publish_rate", 2.0); //Cant be changed during simulation
            this->declare_parameter("tf_lookup_timeout", 0.30);

            const std::string input_topic = this->get_parameter("input_topic").as_string();
            const std::string map_topic = this->get_parameter("map_topic").as_string();
            
            //subscription
            ground_points_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                input_topic,
                10,
                [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){this->slopeCostmapCallback(msg);}
            );
            //publisher
            rclcpp::QoS map_qos(rclcpp::KeepLast(1));
            map_qos.reliable();
            map_qos.transient_local();

            map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic, map_qos);
            //timer
            const double publish_rate = this->get_parameter("publish_rate").as_double();
            const double timer_period = 1.0 / std::max(publish_rate, 0.1);

            map_timer_ = this->create_wall_timer(
                std::chrono::duration<double>(timer_period),
                [this](){this->publishMap();}
            );

            //tf2 components
            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
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

        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr ground_points_sub_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
        rclcpp::TimerBase::SharedPtr map_timer_;

        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        std::unordered_map<CellIndex, double, CellIndexHash> cell_height_;
        std::unordered_map<CellIndex, int, CellIndexHash> cell_confidence_;
        std::unordered_map<CellIndex, double, CellIndexHash> cell_outlier_height_;
        std::unordered_map<CellIndex, int, CellIndexHash> cell_outlier_count_;
        std::vector<int8_t> persistent_costmap_;
        int clouds_processed_ = 0;

        void slopeCostmapCallback(sensor_msgs::msg::PointCloud2::ConstSharedPtr msg){

            const std::string target_frame = this->get_parameter("target_frame").as_string();
            const double grid_resolution = this->get_parameter("grid_resolution").as_double();
            const double map_width = this->get_parameter("map_width").as_double();
            const double map_height = this->get_parameter("map_height").as_double();
            const double origin_x = this->get_parameter("origin_x").as_double();
            const double origin_y = this->get_parameter("origin_y").as_double();
            const double min_insert_range = this->get_parameter("min_insert_range").as_double();
            const double max_insert_range = this->get_parameter("max_insert_range").as_double();
            const double height_alpha = this->get_parameter("height_alpha").as_double();
            const double max_height_update_jump = this->get_parameter("max_height_update_jump").as_double();
            const int outlier_accept_count = this->get_parameter("outlier_accept_count").as_int();
            const double outlier_same_height_tolerance = this->get_parameter("outlier_same_height_tolerance").as_double();
            const double outlier_alpha = this->get_parameter("outlier_alpha").as_double();
            const int max_cell_confidence = this->get_parameter("max_cell_confidence").as_int();
            const double max_batch_z_std = this->get_parameter("max_batch_z_std").as_double();
            const double max_batch_z_range = this->get_parameter("max_batch_z_range").as_double();
            const double height_percentile = this->get_parameter("height_percentile").as_double();
            const double tf_lookup_timeout = this->get_parameter("tf_lookup_timeout").as_double();

            if(msg->header.frame_id.empty()){
                RCLCPP_WARN(this->get_logger(), "Empty frame id");
                return;
            }
            const std::string source_frame = msg->header.frame_id;

            if(grid_resolution <= 0.0) {
                RCLCPP_WARN(this->get_logger(), "Invalid grid_resolution");
                return;
            }

            geometry_msgs::msg::TransformStamped transform_stamped;

            try{
                transform_stamped = tf_buffer_->lookupTransform(
                    target_frame, source_frame,
                    rclcpp::Time(msg->header.stamp),
                    rclcpp::Duration::from_seconds(tf_lookup_timeout)
                );
            }
            catch(const tf2::TransformException& ex){
                RCLCPP_WARN(
                    this->get_logger(),
                    "TF unavailable: %s <- %s: %s",
                    target_frame.c_str(),
                    source_frame.c_str(),
                    ex.what()
                );
                return;
            }

            tf2::Transform source_to_target_transform;
            tf2::fromMsg(transform_stamped.transform, source_to_target_transform);

            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_input(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud_input);

            std::unordered_map<CellIndex, std::vector<double>, CellIndexHash> batch_cell_z;

            int width_cells = static_cast<int>(std::round(map_width / grid_resolution));
            int height_cells = static_cast<int>(std::round(map_height / grid_resolution));

            const double min_range_sq = min_insert_range*min_insert_range;
            const double max_range_sq = max_insert_range*max_insert_range;

            for(const auto& point : cloud_input->points){
                if(!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
                //range filter
                const double range_sq = point.x*point.x + point.y*point.y + point.z*point.z;
                if(range_sq < min_range_sq || range_sq > max_range_sq) continue;
                
                //Apply transform to get point in map frame
                const tf2::Vector3 point_source(point.x, point.y, point.z);
                const tf2::Vector3 point_target = source_to_target_transform * point_source;
                
                const double x = point_target.x();
                const double y = point_target.y();
                const double z = point_target.z();
                
                int ix = static_cast<int>(std::floor((x - origin_x) / grid_resolution));
                int iy = static_cast<int>(std::floor((y - origin_y) / grid_resolution));

                //Out of bounds check
                if(ix < 0 || ix >= width_cells || iy <0 || iy >= height_cells) continue;

                CellIndex index{ix, iy};
                batch_cell_z[index].push_back(z);
            }

            if(batch_cell_z.empty()) return;

            for(const auto& [index, z_values] : batch_cell_z){
                if (z_values.empty()) continue;

                double min_z = std::numeric_limits<double>::infinity();
                double max_z = -std::numeric_limits<double>::infinity();
                
                double sum = 0;
                for(const auto& z : z_values){
                    if(z < min_z) min_z = z;
                    if(z > max_z) max_z = z;
                    sum += z;
                }
                
                double mean = sum / z_values.size();
                double sum_sq = 0;

                for(const auto& z: z_values){
                    sum_sq += (z - mean) * (z - mean);
                }

                const double z_std = std::sqrt(sum_sq / z_values.size());
                const double z_range = max_z - min_z;

                if(z_std > max_batch_z_std) continue;
                if(z_range > max_batch_z_range) continue;

                const double measured_height = percentile(z_values, height_percentile);

                auto height_it = cell_height_.find(index);

                if(height_it == cell_height_.end()){
                    cell_height_[index] = measured_height;
                    cell_confidence_[index] = 1;
                    cell_outlier_height_.erase(index);
                    cell_outlier_count_.erase(index);
                    continue;
                }

                const double old_height = height_it->second;
                const double dz = std::abs(measured_height - old_height);

                if(dz <= max_height_update_jump){
                    const double new_height = (1.0 - height_alpha) * old_height + height_alpha * measured_height;

                    cell_height_[index] = new_height;

                    const int old_confidence = cell_confidence_.count(index) ? cell_confidence_[index] :  0;
                    cell_confidence_[index] = std::min(old_confidence + 1, max_cell_confidence);

                    cell_outlier_height_.erase(index);
                    cell_outlier_count_.erase(index);

                    continue;
                }

                auto outlier_height_it = cell_outlier_height_.find(index);

                if(outlier_height_it == cell_outlier_height_.end()){
                    cell_outlier_height_[index] = measured_height;
                    cell_outlier_count_[index] = 1;
                    continue;
                }

                const double previous_outlier_height = outlier_height_it->second;
                const double outlier_dz = std::abs(measured_height - previous_outlier_height);

                if(outlier_dz <= outlier_same_height_tolerance){
                    const int count = cell_outlier_count_[index] + 1;
                    cell_outlier_count_[index] = count;

                    cell_outlier_height_[index] = 0.8 * previous_outlier_height + 0.2 * measured_height;

                    if(count >= outlier_accept_count){
                        const double accepted_height = cell_outlier_height_[index];

                        const double new_height = (1.0 - outlier_alpha) * old_height + outlier_alpha * accepted_height;
                        cell_height_[index] = new_height;

                        const int old_confidence = cell_confidence_.count(index) ? cell_confidence_[index] : 0;
                        cell_confidence_[index] = std::min(old_confidence + 1, max_cell_confidence);                 
                        cell_outlier_height_.erase(index);
                        cell_outlier_count_.erase(index);
                    }

                    continue;
                }

                cell_outlier_height_[index] = measured_height;
                cell_outlier_count_[index] = 1;
            }
            
            clouds_processed_++;
        }

        void publishMap(){
            const std::string target_frame = this->get_parameter("target_frame").as_string();

            const double grid_resolution = this->get_parameter("grid_resolution").as_double();
            const double map_width = this->get_parameter("map_width").as_double();
            const double map_height = this->get_parameter("map_height").as_double();
            const double origin_x = this->get_parameter("origin_x").as_double();
            const double origin_y = this->get_parameter("origin_y").as_double();
            const int min_cell_confidence = this->get_parameter("min_cell_confidence").as_int();
            const int slope_radius = this->get_parameter("slope_radius").as_int();
            const int min_neighbors_for_slope = this->get_parameter("min_neighbors_for_slope").as_int();
            const double min_neighbor_fill_ratio = this->get_parameter("min_neighbor_fill_ratio").as_double();
            const double max_neighbor_height_range = this->get_parameter("max_neighbor_height_range").as_double();
            const double max_neighbor_height_std = this->get_parameter("max_neighbor_height_std").as_double();
            const double slope_safe = this->get_parameter("slope_safe").as_double();
            const double slope_caution = this->get_parameter("slope_caution").as_double();
            const double slope_danger = this->get_parameter("slope_danger").as_double();
            const int cost_safe = this->get_parameter("cost_safe").as_int();
            const int cost_caution = this->get_parameter("cost_caution").as_int();
            const int cost_danger = this->get_parameter("cost_danger").as_int();
            const int cost_lethal = this->get_parameter("cost_lethal").as_int();
            const int cost_unknown = this->get_parameter("cost_unknown").as_int();
            const double cost_smoothing_alpha_up = this->get_parameter("cost_smoothing_alpha_up").as_double();
            const double cost_smoothing_alpha_down = this->get_parameter("cost_smoothing_alpha_down").as_double();
            const bool enable_cost_decay = this->get_parameter("enable_cost_decay").as_bool();
            const int cost_decay_per_publish = this->get_parameter("cost_decay_per_publish").as_int();
            const bool enable_hole_filling = this->get_parameter("enable_hole_filling").as_bool();
            const int hole_fill_min_known_neighbors = this->get_parameter("hole_fill_min_known_neighbors").as_int();
            const int warmup_clouds = this->get_parameter("warmup_clouds").as_int();
            const bool enable_publish_during_warmup = this->get_parameter("enable_publish_during_warmup").as_bool();
        }

        double percentile(std::vector<double> values, double percentile_value){
            if(values.empty()){
                return 0.0;
            }

            percentile_value = std::clamp(percentile_value, 0.0, 100.0);
            std::sort(values.begin(), values.end());

            if(values.size() == 1){
                return values.front();
            }

            const double position = (percentile_value / 100.0) * static_cast<double>(values.size() - 1);
            const std::size_t index = static_cast<std::size_t>(std::floor(position));
            
            return values[index];
        }
};
