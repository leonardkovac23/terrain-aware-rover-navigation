#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <stdexcept>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>

#include "onnxruntime_cxx_api.h"

class TerrainSegmentationNode : public rclcpp::Node{
    public:
        TerrainSegmentationNode() : rclcpp::Node("terrain_segmentation_node"), 
        ort_env_(ORT_LOGGING_LEVEL_WARNING, "terrain_segmentation_node"),
        frame_count_(0){
            //ROS image interface
            this->declare_parameter("image_topic", "/rgbd_camera/image");
            this->declare_parameter("mask_topic", "/terrain_segmentation/mask");
            this->declare_parameter("color_mask_topic", "/terrain_segmentation/color_mask");
            this->declare_parameter("overlay_topic", "/terrain_segmentation/overlay");
            //ONNX model input setup
            this->declare_parameter("model_path", "dataset/models_onnx/terrain_segmentation/ddrnet_39.onnx");
            this->declare_parameter("input_size", 256);
            this->declare_parameter("num_classes", 5);
            //Runtime behavior and visualization output
            this->declare_parameter("publish_every_n", 1);
            this->declare_parameter("threads", 1);
            this->declare_parameter("publish_color_mask", true);
            this->declare_parameter("overlay_alpha", 0.35);

            const std::string image_topic = this->get_parameter("image_topic").as_string();
            const std::string mask_topic = this->get_parameter("mask_topic").as_string();
            const std::string color_mask_topic = this->get_parameter("color_mask_topic").as_string();
            const std::string overlay_topic = this->get_parameter("overlay_topic").as_string();

            const std::string model_path = this->get_parameter("model_path").as_string();
            input_size_ = static_cast<int>(this->get_parameter("input_size").as_int());
            num_classes_ = static_cast<int>(this->get_parameter("num_classes").as_int());
            publish_every_n_ = static_cast<int>(std::max<int64_t>(1, this->get_parameter("publish_every_n").as_int()));
            threads_ = static_cast<int>(std::max<int64_t>(1, this->get_parameter("threads").as_int()));
            publish_color_mask_ = this->get_parameter("publish_color_mask").as_bool();
            overlay_alpha_ = this->get_parameter("overlay_alpha").as_double();
            overlay_alpha_ = std::clamp(overlay_alpha_, 0.0, 1.0);

            if(input_size_ <= 0){
                throw std::runtime_error("input_size must be positive");
            }

            if(num_classes_ <= 0){
                throw std::runtime_error("num_classes must be positive");
            }

            //ONNX session setup
            session_options_.SetIntraOpNumThreads(threads_);
            session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            session_ = std::make_unique<Ort::Session>(ort_env_, model_path.c_str(), session_options_);
            
            auto input_name = session_->GetInputNameAllocated(0, allocator_);
            auto output_name = session_->GetOutputNameAllocated(0, allocator_);

            input_name_ = input_name.get();
            output_name_ = output_name.get();

            input_shape_ = {1, 3, input_size_, input_size_};
            //Subscription
            image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
                image_topic,
                10,
                [this](sensor_msgs::msg::Image::ConstSharedPtr msg){this->imageCallback(msg);}
            );

            //Publisher
            mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>(mask_topic, 10);
            color_mask_pub_ = this->create_publisher<sensor_msgs::msg::Image>(color_mask_topic, 10);
            overlay_pub_ = this->create_publisher<sensor_msgs::msg::Image>(overlay_topic, 10);

            RCLCPP_INFO(this->get_logger(), "terrain_segmentation_node started.");
            RCLCPP_INFO(this->get_logger(), "Loaded ONNX model: %s", model_path.c_str());
            RCLCPP_INFO(this->get_logger(), "ONNX Runtime threads: %d", threads_);
            RCLCPP_INFO(this->get_logger(), "Input name: %s | output name: %s", input_name_.c_str(), output_name_.c_str());
            RCLCPP_INFO(this->get_logger(), "Subscribing: %s", image_topic.c_str());
            RCLCPP_INFO(this->get_logger(), "Publishing mask: %s", mask_topic.c_str());
            if(publish_color_mask_){
                RCLCPP_INFO(this->get_logger(), "Publishing color mask: %s", color_mask_topic.c_str());
            }
            RCLCPP_INFO(this->get_logger(), "Publishing overlay: %s", overlay_topic.c_str());
        }

    private:
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr color_mask_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;

        Ort::Env ort_env_;
        Ort::SessionOptions session_options_;
        std::unique_ptr<Ort::Session> session_;
        Ort::AllocatorWithDefaultOptions allocator_;

        std::string input_name_;
        std::string output_name_;

        std::vector<int64_t> input_shape_;

        int input_size_;
        int num_classes_;
        int publish_every_n_;
        int threads_;
        int frame_count_;
        bool publish_color_mask_;
        double overlay_alpha_;

        const uint8_t background_class_id_ = 0;
        const std::vector<cv::Vec3b> terrain_class_colors_bgr_{
            cv::Vec3b(0, 255, 0),
            cv::Vec3b(0, 255, 255),
            cv::Vec3b(255, 0, 255),
            cv::Vec3b(255, 220, 80)
        };

        void imageCallback(sensor_msgs::msg::Image::ConstSharedPtr msg){
            frame_count_++;
            if(frame_count_ % publish_every_n_ != 0){
                return;
            }

            try{
                cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
                const cv::Mat& bgr_image = cv_ptr->image;

                if(bgr_image.empty()){
                    RCLCPP_WARN(this->get_logger(), "Received empty image");
                    return;
                }

                std::vector<float> input_tensor_values = preprocessImage(bgr_image);
                const std::vector<float> logits = runInference(input_tensor_values);
                cv::Mat mask = logitsToMask(logits);

                if(mask.size() != bgr_image.size()){
                    cv::resize(mask, mask, bgr_image.size(), 0.0, 0.0, cv::INTER_NEAREST);
                }

                cv_bridge::CvImage mask_msg;
                mask_msg.header = msg->header;
                mask_msg.encoding = "mono8";
                mask_msg.image = mask;
                mask_pub_->publish(*mask_msg.toImageMsg());

                const cv::Mat color_mask = colorizeMask(mask);

                if(publish_color_mask_){
                    cv_bridge::CvImage color_msg;
                    color_msg.header = msg->header;
                    color_msg.encoding = "bgr8";
                    color_msg.image = color_mask;
                    color_mask_pub_->publish(*color_msg.toImageMsg());
                }

                cv::Mat overlay = bgr_image.clone();
                cv::Mat blended;
                cv::addWeighted(
                    bgr_image,
                    1.0 - overlay_alpha_,
                    color_mask,
                    overlay_alpha_,
                    0.0,
                    blended
                );
                blended.copyTo(overlay, mask != background_class_id_);

                cv_bridge::CvImage overlay_msg;
                overlay_msg.header = msg->header;
                overlay_msg.encoding = "bgr8";
                overlay_msg.image = overlay;
                overlay_pub_->publish(*overlay_msg.toImageMsg());
            }
            catch(const cv_bridge::Exception& exc){
                RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed: %s", exc.what());
                return;
            }
            catch(const Ort::Exception& exc){
                RCLCPP_ERROR(this->get_logger(), "ONNX inference failed: %s", exc.what());
                return;
            }
            catch(const std::exception& exc){
                RCLCPP_ERROR(this->get_logger(), "Terrain segmentation failed: %s", exc.what());
                return;
            }
        }

        std::vector<float> preprocessImage(const cv::Mat& bgr_image){
            cv::Mat resized;
            cv::resize(bgr_image, resized, cv::Size(input_size_, input_size_), 0.0, 0.0, cv::INTER_LINEAR);

            cv::Mat rgb;
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

            cv::Mat rgb_float;
            rgb.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

            const std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
            const std::array<float, 3> std{0.229F, 0.224F, 0.225F};

            std::vector<float> input_tensor_values(3 * input_size_ * input_size_);
            //HWC -> CHW
            for(int y = 0; y < input_size_; ++y){
                for(int x = 0; x < input_size_; ++x){
                    const cv::Vec3f pixel = rgb_float.at<cv::Vec3f>(y, x);

                    for(int channel = 0; channel < 3; ++channel){
                        const float normalized = (pixel[channel] - mean[channel]) / std[channel];
                        const int tensor_index = channel * input_size_ * input_size_ + y * input_size_ + x;
                        input_tensor_values[tensor_index] = normalized;
                    }
                }
            }

            return input_tensor_values;
        }

        std::vector<float> runInference(std::vector<float>& input_tensor_values){
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_shape_.data(),
                input_shape_.size()
            );

            const char* input_names[] = {input_name_.c_str()};
            const char* output_names[] = {output_name_.c_str()};

            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            float* output_data = output_tensors.front().GetTensorMutableData<float>();
            auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

            std::size_t output_size = 1;
            
            for(const auto dim : output_shape){
                output_size *= static_cast<std::size_t>(dim);
            }

            return std::vector<float>(output_data, output_data + output_size);

        }

        cv::Mat logitsToMask(const std::vector<float>& logits){
            const int image_area = input_size_ * input_size_;
            const std::size_t expected_size = static_cast<std::size_t>(num_classes_) * static_cast<std::size_t>(image_area);

            if(logits.size() != expected_size){
                throw std::runtime_error("Unexpected logits size from ONNX model");
            }

            cv::Mat mask(input_size_, input_size_, CV_8UC1);

            for(int y = 0; y < input_size_; ++y){
                for(int x = 0; x < input_size_; ++x){
                    const int pixel_index = y * input_size_ + x;

                    int best_class = 0;
                    float best_logit = logits[pixel_index];

                    for(int class_id = 1; class_id < num_classes_; ++class_id){
                        const int logit_index = class_id * image_area + pixel_index;
                        const float logit = logits[logit_index];

                        if(logit > best_logit){
                            best_logit = logit;
                            best_class = class_id;
                        }
                    }

                    mask.at<uint8_t>(y, x) = static_cast<uint8_t>(best_class);
                }
            }

            return mask;
        }

        cv::Mat colorizeMask(const cv::Mat& mask){
            cv::Mat color_mask(mask.rows, mask.cols, CV_8UC3, cv::Scalar(0, 0, 0));

            for(int y = 0; y < mask.rows; ++y){
                for(int x = 0; x < mask.cols; ++x){
                    const int class_id = static_cast<int>(mask.at<uint8_t>(y, x));

                    if(class_id == background_class_id_) continue;

                    const int color_index = class_id - 1;
                    if(color_index >= 0 && color_index < static_cast<int>(terrain_class_colors_bgr_.size())){
                        color_mask.at<cv::Vec3b>(y, x) = terrain_class_colors_bgr_[color_index];
                    }
                }
            }

            return color_mask;
        }

};

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TerrainSegmentationNode>());
    rclcpp::shutdown();

    return 0;
}
