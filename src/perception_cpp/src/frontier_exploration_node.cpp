#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"

#include "tf2/exceptions.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

class FrontierExplorationNode : public rclcpp::Node{
    public:
        using NavigateToPose = nav2_msgs::action::NavigateToPose;
        using GoalHandleNavigate = rclcpp_action::ClientGoalHandle<NavigateToPose>;

        FrontierExplorationNode() : rclcpp::Node("frontier_exploration_node"){
            //ROS interface and frame setup//
            this->declare_parameter("map_topic", "/map/traversability");
            this->declare_parameter("target_frame", "map");
            this->declare_parameter("robot_frame", "base_link");
            this->declare_parameter("navigate_action_name", "navigate_to_pose");
            //Frontier candidate filters
            this->declare_parameter("unknown_cost", -1);
            this->declare_parameter("max_goal_cost", 70);
            this->declare_parameter("lethal_cost_threshold", 95);
            this->declare_parameter("min_goal_distance", 1.0);
            this->declare_parameter("goal_reached_distance", 0.70);
            this->declare_parameter("obstacle_clearance_radius_cells", 2);
            this->declare_parameter("min_unknown_neighbors", 1);
            //Goal scoring
            this->declare_parameter("information_gain_radius_cells", 5);
            this->declare_parameter("information_gain_weight", 2.0);
            this->declare_parameter("distance_weight", 1.0);
            this->declare_parameter("cost_weight", 0.8);
            //Runtime behavior
            this->declare_parameter("planning_rate", 0.5); //Cant be changed during simulation
            this->declare_parameter("goal_cooldown", 3.0);
            this->declare_parameter("tf_lookup_timeout", 0.30);

            const std::string map_topic = this->get_parameter("map_topic").as_string();
            const std::string navigate_action_name = this->get_parameter("navigate_action_name").as_string();

            //Map QoS
            rclcpp::QoS map_qos(rclcpp::KeepLast(1));
            map_qos.reliable();
            map_qos.transient_local();

            //subscription
            map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
                map_topic,
                map_qos,
                [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg){this->mapCallback(msg);}
            );

            //navigation
            navigate_client_ = rclcpp_action::create_client<NavigateToPose>(this, navigate_action_name);

            const double planning_rate = this->get_parameter("planning_rate").as_double();
            const double timer_period = 1.0 / std::max(planning_rate, 0.1);

            planning_timer_ = this->create_wall_timer(
                std::chrono::duration<double>(timer_period),
                [this](){this->planningTimerCallback();}
            );

            tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            RCLCPP_INFO(this->get_logger(), "frontier_exploration_node started.");
            RCLCPP_INFO(this->get_logger(), "Subscribing map: %s", map_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "Using action: %s", navigate_action_name.c_str());
        }

    private:
        struct FrontierCandidate{
            int x;
            int y;
            double world_x;
            double world_y;
            double score;
            int unknown_neighbors;
            int information_gain;
            int cost;
        };

        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
        rclcpp_action::Client<NavigateToPose>::SharedPtr navigate_client_;
        rclcpp::TimerBase::SharedPtr planning_timer_;

        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

        nav_msgs::msg::OccupancyGrid::ConstSharedPtr latest_map_;
        geometry_msgs::msg::PoseStamped active_goal_;
        rclcpp::Time last_goal_sent_;
        bool goal_active_ = false;

        void mapCallback(nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg){
            latest_map_ = msg;
        }

        void planningTimerCallback(){
            if(!latest_map_){
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for traversability map");
                return;
            }

            const std::string target_frame = this->get_parameter("target_frame").as_string();
            const std::string robot_frame = this->get_parameter("robot_frame").as_string();
            const double tf_lookup_timeout = this->get_parameter("tf_lookup_timeout").as_double();

            geometry_msgs::msg::TransformStamped robot_transform;

            try{
                robot_transform = tf_buffer_->lookupTransform(
                    target_frame,
                    robot_frame,
                    rclcpp::Time(0, 0, this->get_clock()->get_clock_type()),
                    rclcpp::Duration::from_seconds(tf_lookup_timeout)
                );
            }
            catch(const tf2::TransformException& ex){
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    2000,
                    "TF unavailable: %s <- %s: %s",
                    target_frame.c_str(),
                    robot_frame.c_str(),
                    ex.what()
                );
                return;
            }

            const double robot_x = robot_transform.transform.translation.x;
            const double robot_y = robot_transform.transform.translation.y;

            if(goal_active_){
                const double distance_to_goal = std::hypot(active_goal_.pose.position.x - robot_x, active_goal_.pose.position.y - robot_y);

                const double goal_reached_distance = this->get_parameter("goal_reached_distance").as_double();

                if(distance_to_goal > goal_reached_distance) return;

                goal_active_ = false;
            }

            const rclcpp::Time now = this->get_clock()->now();
            const double goal_cooldown = this->get_parameter("goal_cooldown").as_double();

            if(last_goal_sent_.nanoseconds() > 0 && now - last_goal_sent_ < rclcpp::Duration::from_seconds(goal_cooldown)) return;

            FrontierCandidate candidate;
            if(!findBestFrontier(robot_x, robot_y, candidate)){
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000, "No frontier goal found");
                return;
            }

            sendGoal(candidate, robot_x, robot_y);
        }

        bool findBestFrontier(double robot_x, double robot_y, FrontierCandidate& best_candidate){
            const auto& map = *latest_map_;

            if(map.data.empty()) return false;
            if(map.info.resolution <= 0.0) return false;

            const int width_cells = static_cast<int>(map.info.width);
            const int height_cells = static_cast<int>(map.info.height);

            if(width_cells <= 0 || height_cells <= 0) return false;

            const int unknown_cost = static_cast<int>(this->get_parameter("unknown_cost").as_int());
            const int max_goal_cost = std::clamp(static_cast<int>(this->get_parameter("max_goal_cost").as_int()), 0, 100);
            const int min_unknown_neighbors = static_cast<int>(std::max<int64_t>(1, this->get_parameter("min_unknown_neighbors").as_int()));
            const int information_gain_radius_cells = static_cast<int>(std::max<int64_t>(1, this->get_parameter("information_gain_radius_cells").as_int()));
            const double information_gain_weight = this->get_parameter("information_gain_weight").as_double();
            const double distance_weight = this->get_parameter("distance_weight").as_double();
            const double cost_weight = this->get_parameter("cost_weight").as_double();
            const double min_goal_distance = this->get_parameter("min_goal_distance").as_double();

            bool found_candidate = false;
            double best_score = -std::numeric_limits<double>::infinity();

            for(int y = 1; y < height_cells - 1; ++y){
                for(int x = 1; x < width_cells - 1; ++x){
                    const int index = y * width_cells + x;
                    const int cost = static_cast<int>(map.data[index]);

                    if(cost < 0 || cost > max_goal_cost) continue;
                    if(hasObstacleTooClose(map, x, y)) continue;
                    
                    const int unknown_neighbors = countUnknownNeighbors(map, x, y, 1, unknown_cost);
                    if(unknown_neighbors < min_unknown_neighbors) continue;

                    const double world_x = map.info.origin.position.x + (static_cast<double>(x) + 0.5) * map.info.resolution;
                    const double world_y = map.info.origin.position.y + (static_cast<double>(y) + 0.5) * map.info.resolution;
                    const double distance = std::hypot(world_x - robot_x, world_y - robot_y);

                    if(distance < min_goal_distance) continue;

                    const int information_gain = countUnknownNeighbors(map, x, y, information_gain_radius_cells, unknown_cost);
                    const double score =
                        information_gain_weight * static_cast<double>(information_gain) -
                        distance_weight * distance -
                        cost_weight * static_cast<double>(cost);

                    if(score <= best_score) continue;

                    best_score = score;
                    best_candidate = FrontierCandidate{
                        x,
                        y,
                        world_x,
                        world_y,
                        score,
                        unknown_neighbors,
                        information_gain,
                        cost
                    };
                    found_candidate = true;
                }
            }

            return found_candidate;
        }

        bool hasObstacleTooClose(const nav_msgs::msg::OccupancyGrid& map, int cell_x, int cell_y){
            const int width_cells = static_cast<int>(map.info.width);
            const int height_cells = static_cast<int>(map.info.height);
            const int radius = static_cast<int>(std::max<int64_t>(0, this->get_parameter("obstacle_clearance_radius_cells").as_int()));
            const int lethal_cost_threshold = std::clamp(static_cast<int>(this->get_parameter("lethal_cost_threshold").as_int()), 0, 100);

            if(radius <= 0) return false;

            for(int dy = -radius; dy <= radius; ++dy){
                for(int dx = -radius; dx <= radius; ++dx){
                    const int x = cell_x + dx;
                    const int y = cell_y + dy;

                    if(x < 0 || x >= width_cells || y < 0 || y >= height_cells) continue;
                    if(dx * dx + dy * dy > radius * radius) continue;

                    const int index = y * width_cells + x;
                    const int cost = static_cast<int>(map.data[index]);

                    if(cost >= lethal_cost_threshold) return true;
                }
            }

            return false;
        }

        int countUnknownNeighbors(const nav_msgs::msg::OccupancyGrid& map, int cell_x, int cell_y, int radius, int unknown_cost){
            const int width_cells = static_cast<int>(map.info.width);
            const int height_cells = static_cast<int>(map.info.height);
            int count = 0;

            for(int dy = -radius; dy <= radius; ++dy){
                for(int dx = -radius; dx <= radius; ++dx){
                    if(dx == 0 && dy == 0) continue;

                    const int x = cell_x + dx;
                    const int y = cell_y + dy;

                    if(x < 0 || x >= width_cells || y < 0 || y >= height_cells) continue;

                    const int index = y * width_cells + x;
                    const int cost = static_cast<int>(map.data[index]);

                    if(cost == unknown_cost) count++;
                }
            }

            return count;
        }

        void sendGoal(const FrontierCandidate& candidate, double robot_x, double robot_y){
            using namespace std::chrono_literals;

            if(!navigate_client_->wait_for_action_server(0s)){
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Nav2 NavigateToPose action server is not ready");
                return;
            }

            NavigateToPose::Goal goal_msg;
            goal_msg.pose.header.stamp = this->get_clock()->now();
            goal_msg.pose.header.frame_id = this->get_parameter("target_frame").as_string();
            goal_msg.pose.pose.position.x = candidate.world_x;
            goal_msg.pose.pose.position.y = candidate.world_y;
            goal_msg.pose.pose.position.z = 0.0;

            const double yaw = std::atan2(candidate.world_y - robot_y, candidate.world_x - robot_x);
            tf2::Quaternion orientation;
            orientation.setRPY(0.0, 0.0, yaw);
            goal_msg.pose.pose.orientation = tf2::toMsg(orientation);

            auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

            send_goal_options.goal_response_callback =
                [this](const GoalHandleNavigate::SharedPtr& goal_handle){
                    if(!goal_handle){
                        goal_active_ = false;
                        RCLCPP_WARN(this->get_logger(), "Exploration goal rejected");
                    }
                };

            send_goal_options.result_callback =
                [this](const GoalHandleNavigate::WrappedResult& result){
                    (void)result;
                    goal_active_ = false;
                    RCLCPP_INFO(this->get_logger(), "Exploration goal finished");
                };

            navigate_client_->async_send_goal(goal_msg, send_goal_options);

            active_goal_ = goal_msg.pose;
            last_goal_sent_ = this->get_clock()->now();
            goal_active_ = true;

            RCLCPP_INFO(
                this->get_logger(),
                "Sent exploration goal | cell: %d,%d | world: %.2f,%.2f | score: %.1f | unknown gain: %d | cost: %d",
                candidate.x,
                candidate.y,
                candidate.world_x,
                candidate.world_y,
                candidate.score,
                candidate.information_gain,
                candidate.cost
            );
        }
};

int main(int argc, char ** argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FrontierExplorationNode>());
    rclcpp::shutdown();
    return 0;
}
