#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"

class TraversabilityCostmapNode : public rclcpp::Node{
    public:
        TraversabilityCostmapNode() : rclcpp::Node("traversability_costmap_node"){
            //ROS interface
            this->declare_parameter("slope_map_topic", "/map/slope");
            this->declare_parameter("obstacle_map_topic", "/map/obstacles");
            this->declare_parameter("semantic_map_topic", "/map/terrain_semantic");
            this->declare_parameter("map_topic", "/map/traversability");
            //Map merge policy
            this->declare_parameter("use_semantic_terrain_map", false);
            this->declare_parameter("slope_weight", 0.75);
            this->declare_parameter("semantic_weight", 0.25);
            this->declare_parameter("risk_preserve_ratio", 0.75);
            this->declare_parameter("obstacle_lethal_threshold", 95);
            this->declare_parameter("obstacle_output_cost", 100);
            this->declare_parameter("unknown_cost", -1);
            this->declare_parameter("require_matching_geometry", true);
            //Virtual terrain boundary in target frame
            this->declare_parameter("enable_boundary_safety", true);
            this->declare_parameter("boundary_min_x", -20.0);
            this->declare_parameter("boundary_max_x", 20.0);
            this->declare_parameter("boundary_min_y", -20.0);
            this->declare_parameter("boundary_max_y", 20.0);
            this->declare_parameter("boundary_margin", 1.0);
            this->declare_parameter("boundary_cost", 100);
            //Runtime behavior
            this->declare_parameter("publish_rate", 2.0); //Cant be changed during simulation

            const std::string slope_map_topic = this->get_parameter("slope_map_topic").as_string();
            const std::string obstacle_map_topic = this->get_parameter("obstacle_map_topic").as_string();
            const std::string semantic_map_topic = this->get_parameter("semantic_map_topic").as_string();
            const std::string map_topic = this->get_parameter("map_topic").as_string();
            const bool use_semantic_terrain_map = this->get_parameter("use_semantic_terrain_map").as_bool();

            rclcpp::QoS map_qos(rclcpp::KeepLast(1));
            map_qos.reliable();
            map_qos.transient_local();

            //Subscriptions
            slope_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
                slope_map_topic,
                map_qos,
                [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg){this->slopeMapCallback(msg);}
            );
            obstacle_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
                obstacle_map_topic,
                map_qos,
                [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg){this->obstacleMapCallback(msg);}
            );
            semantic_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
                semantic_map_topic,
                map_qos,
                [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg){this->semanticMapCallback(msg);}
            );
            //Publisher
            map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic, map_qos);

            const double publish_rate = this->get_parameter("publish_rate").as_double();
            const double timer_period = 1.0 / std::max(publish_rate, 0.1);

            map_timer_ = this->create_wall_timer(
                std::chrono::duration<double>(timer_period),
                [this](){this->publishMap();}
            );

            RCLCPP_INFO(this->get_logger(), "traversability_costmap_node started.");
            RCLCPP_INFO(this->get_logger(), "Subscribing slope map: %s", slope_map_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "Subscribing obstacle map: %s", obstacle_map_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "Subscribing semantic map: %s", semantic_map_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "Using semantic terrain map: %s", use_semantic_terrain_map ? "true" : "false");
            RCLCPP_INFO(this->get_logger(), "Publishing: %s", map_topic.c_str());
        }

    private:
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr slope_map_sub_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr obstacle_map_sub_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr semantic_map_sub_;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;
        rclcpp::TimerBase::SharedPtr map_timer_;

        nav_msgs::msg::OccupancyGrid::ConstSharedPtr latest_slope_map_;
        nav_msgs::msg::OccupancyGrid::ConstSharedPtr latest_obstacle_map_;
        nav_msgs::msg::OccupancyGrid::ConstSharedPtr latest_semantic_map_;

        void slopeMapCallback(nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg){
            latest_slope_map_ = msg;
        }

        void obstacleMapCallback(nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg){
            latest_obstacle_map_ = msg;
        }

        void semanticMapCallback(nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg){
            latest_semantic_map_ = msg;
        }

        void publishMap(){
            if(!latest_slope_map_){
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for slope map");
                return;
            }

            if(!latest_obstacle_map_){
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for obstacle map");
                return;
            }

            const bool require_matching_geometry = this->get_parameter("require_matching_geometry").as_bool();
            const bool use_semantic_terrain_map = this->get_parameter("use_semantic_terrain_map").as_bool();

            if(use_semantic_terrain_map && !latest_semantic_map_){
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for semantic terrain map");
                return;
            }

            if(require_matching_geometry && !mapsHaveSameGeometry(*latest_slope_map_, *latest_obstacle_map_)){
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    2000,
                    "Slope and obstacle maps do not have matching geometry"
                );
                return;
            }

            if(use_semantic_terrain_map && require_matching_geometry && !mapsHaveSameGeometry(*latest_slope_map_, *latest_semantic_map_)){
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    2000,
                    "Slope and semantic terrain maps do not have matching geometry"
                );
                return;
            }

            const std::size_t map_size = latest_slope_map_->data.size();

            if(map_size == 0){
                RCLCPP_WARN(this->get_logger(), "Slope map is empty");
                return;
            }

            if(latest_obstacle_map_->data.size() != map_size){
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Map data sizes do not match");
                return;
            }

            if(use_semantic_terrain_map && latest_semantic_map_->data.size() != map_size){
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Semantic map data size does not match");
                return;
            }

            const int obstacle_lethal_threshold = std::clamp(static_cast<int>(this->get_parameter("obstacle_lethal_threshold").as_int()),0,100);
            const int obstacle_output_cost = std::clamp(static_cast<int>(this->get_parameter("obstacle_output_cost").as_int()),0,100);
            const int unknown_cost = static_cast<int>(this->get_parameter("unknown_cost").as_int());
            const double slope_weight = std::max(0.0, this->get_parameter("slope_weight").as_double());
            const double semantic_weight = std::max(0.0, this->get_parameter("semantic_weight").as_double());
            const double risk_preserve_ratio = std::clamp(this->get_parameter("risk_preserve_ratio").as_double(), 0.0, 1.0);
            const bool enable_boundary_safety = this->get_parameter("enable_boundary_safety").as_bool();
            const double boundary_min_x = this->get_parameter("boundary_min_x").as_double();
            const double boundary_max_x = this->get_parameter("boundary_max_x").as_double();
            const double boundary_min_y = this->get_parameter("boundary_min_y").as_double();
            const double boundary_max_y = this->get_parameter("boundary_max_y").as_double();
            const double boundary_margin = std::max(0.0, this->get_parameter("boundary_margin").as_double());
            const int boundary_cost = std::clamp(static_cast<int>(this->get_parameter("boundary_cost").as_int()),0,100);
            const int width_cells = static_cast<int>(latest_slope_map_->info.width);
            const int height_cells = static_cast<int>(latest_slope_map_->info.height);
            const double resolution = latest_slope_map_->info.resolution;
            const double origin_x = latest_slope_map_->info.origin.position.x;
            const double origin_y = latest_slope_map_->info.origin.position.y;

            if(width_cells <= 0 || height_cells <= 0 || resolution <= 0.0){
                RCLCPP_WARN(this->get_logger(), "Invalid traversability map geometry");
                return;
            }

            std::vector<int8_t> data(map_size, static_cast<int8_t>(unknown_cost));

            int obstacle_cells = 0;
            int unknown_cells = 0;
            int known_cells = 0;
            int boundary_cells = 0;
            int semantic_cells = 0;
            int semantic_raised_cells = 0;

            for(std::size_t index = 0; index < map_size; ++index){
                const int cell_x = static_cast<int>(index % static_cast<std::size_t>(width_cells));
                const int cell_y = static_cast<int>(index / static_cast<std::size_t>(width_cells));

                const double world_x = origin_x + (static_cast<double>(cell_x) + 0.5) * resolution;
                const double world_y = origin_y + (static_cast<double>(cell_y) + 0.5) * resolution;
                
                const int slope_cost = static_cast<int>(latest_slope_map_->data[index]);
                const int obstacle_cost = static_cast<int>(latest_obstacle_map_->data[index]);
                int semantic_cost = -1;

                if(use_semantic_terrain_map){
                    semantic_cost = static_cast<int>(latest_semantic_map_->data[index]);
                }

                if(enable_boundary_safety && isBoundaryCell(world_x, world_y, boundary_min_x, boundary_max_x, boundary_min_y, boundary_max_y, boundary_margin)){
                    data[index] = static_cast<int8_t>(boundary_cost);
                    boundary_cells++;
                    continue;
                }

                if(obstacle_cost >= obstacle_lethal_threshold){
                    data[index] = static_cast<int8_t>(obstacle_output_cost);
                    obstacle_cells++;
                    continue;
                }

                if(slope_cost < 0 && semantic_cost < 0){
                    data[index] = static_cast<int8_t>(unknown_cost);
                    unknown_cells++;
                    continue;
                }

                int final_cost = 0;
                const bool has_slope_cost = slope_cost >= 0;
                const bool has_obstacle_cost = obstacle_cost >= 0;
                const bool has_semantic_cost = semantic_cost >= 0;

                if(has_obstacle_cost){
                    final_cost = std::max(final_cost, std::clamp(obstacle_cost, 0, 100));
                }

                const int cost_without_semantic = final_cost;

                if(has_slope_cost && has_semantic_cost){
                    const int terrain_cost = combineSlopeAndSemanticCost(
                        std::clamp(slope_cost, 0, 100),
                        std::clamp(semantic_cost, 0, 100),
                        slope_weight,
                        semantic_weight,
                        risk_preserve_ratio
                    );

                    final_cost = std::max(final_cost, terrain_cost);
                    semantic_cells++;

                    if(final_cost > std::max(cost_without_semantic, std::clamp(slope_cost, 0, 100))){
                        semantic_raised_cells++;
                    }
                }
                else if(has_slope_cost){
                    final_cost = std::max(final_cost, std::clamp(slope_cost, 0, 100));
                }
                else if(has_semantic_cost){
                    const int clamped_semantic_cost = std::clamp(semantic_cost, 0, 100);
                    final_cost = std::max(final_cost, clamped_semantic_cost);
                    semantic_cells++;

                    if(final_cost > cost_without_semantic){
                        semantic_raised_cells++;
                    }
                }

                data[index] = static_cast<int8_t>(final_cost);
                known_cells++;
            }

            nav_msgs::msg::OccupancyGrid msg = *latest_slope_map_;
            msg.header.stamp = this->get_clock()->now();
            msg.info.map_load_time = this->get_clock()->now();
            msg.data = data;

            map_pub_->publish(msg);

            RCLCPP_INFO_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Published traversability map | known: %d | unknown: %d | obstacles: %d | boundary: %d | semantic used: %d | semantic raised: %d | map: %ux%u",
                known_cells,
                unknown_cells,
                obstacle_cells,
                boundary_cells,
                semantic_cells,
                semantic_raised_cells,
                msg.info.width,
                msg.info.height
            );
        }

        bool mapsHaveSameGeometry(const nav_msgs::msg::OccupancyGrid& first, const nav_msgs::msg::OccupancyGrid& second){
            const double position_epsilon = 1e-6;
            const double resolution_epsilon = 1e-6;

            if(first.header.frame_id != second.header.frame_id) return false;
            if(first.info.width != second.info.width) return false;
            if(first.info.height != second.info.height) return false;
            if(std::abs(first.info.resolution - second.info.resolution) > resolution_epsilon) return false;
            if(std::abs(first.info.origin.position.x - second.info.origin.position.x) > position_epsilon) return false;
            if(std::abs(first.info.origin.position.y - second.info.origin.position.y) > position_epsilon) return false;

            return true;
        }

        int combineSlopeAndSemanticCost(int slope_cost, int semantic_cost, double slope_weight, double semantic_weight, double risk_preserve_ratio){
            const double weight_sum = slope_weight + semantic_weight;
            const int max_risk = std::max(slope_cost, semantic_cost);
            int weighted_cost = max_risk;

            if(weight_sum > 1e-9){
                const double weighted = (slope_weight * static_cast<double>(slope_cost) +
                    semantic_weight * static_cast<double>(semantic_cost)
                ) / weight_sum;

                weighted_cost = static_cast<int>(std::lround(weighted));
            }

            const int risk_floor = static_cast<int>(std::lround(static_cast<double>(max_risk) * risk_preserve_ratio));
            const int final_cost = std::max(weighted_cost, risk_floor);

            return std::clamp(final_cost, 0, 100);
        }

        bool isBoundaryCell(double world_x, double world_y, double min_x, double max_x, double min_y, double max_y, double margin){
            const double safe_min_x = min_x + margin;
            const double safe_max_x = max_x - margin;
            const double safe_min_y = min_y + margin;
            const double safe_max_y = max_y - margin;

            if(safe_min_x >= safe_max_x) return true;
            if(safe_min_y >= safe_max_y) return true;
            if(world_x < safe_min_x) return true;
            if(world_x > safe_max_x) return true;
            if(world_y < safe_min_y) return true;
            if(world_y > safe_max_y) return true;

            return false;
        }
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TraversabilityCostmapNode>());
    rclcpp::shutdown();
    return 0;
}
