#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/search/pcl_search.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

#include <tf2_eigen/tf2_eigen.h>

class ConcatSensorToMap : public rclcpp::Node
{
public:
  ConcatSensorToMap(const rclcpp::NodeOptions & node_options) : Node("concat_sensor_to_map", node_options)
  {
    using std::placeholders::_1;

    leaf_size_ = this->declare_parameter("leaf_size", 0.2);
    min_displacement_ = this->declare_parameter("min_displacement", 5.0);
    min_distance_threshold_ = this->declare_parameter("min_distance_threshold", 0.3);
    base_frame_id_ = this->declare_parameter("base_frame_id", "base_link");
    map_frame_id_ = this->declare_parameter("map_frame_id", "map");
    save_path_ = this->declare_parameter("save_path", "");
    pcd_path_ = this->declare_parameter("pcd_path", "");

    points_subscriber_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "~/input/points_raw", rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&ConcatSensorToMap::callbackPointsCloud, this, _1));

    map_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path_+"/pointcloud_map.pcd", *map_cloud_) == -1) exit(-1);

    if(!tree_) {
      tree_.reset(new pcl::search::KdTree<pcl::PointXYZ>(true));
    }
    tree_->setInputCloud(map_cloud_);
  }
  ~ConcatSensorToMap() = default;

private:
  geometry_msgs::msg::TransformStamped getTransform(
    const std::string target_frame, const std::string source_frame)
  {
    geometry_msgs::msg::TransformStamped frame_transform;
    try {
      frame_transform = tf2_buffer_.lookupTransform(
        target_frame, source_frame, tf2::TimePointZero, tf2::durationFromSec(0.5));
    } catch (tf2::TransformException & ex) {
      RCLCPP_DEBUG(get_logger(), "%s", ex.what());
    }
    return frame_transform;
  }
  void transformPointCloud(
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr & output_ptr,
    const geometry_msgs::msg::TransformStamped frame_transform)
  {
    const Eigen::Affine3d base_to_sensor_frame_affine = tf2::transformToEigen(frame_transform);
    const Eigen::Matrix4f base_to_sensor_frame_matrix =
      base_to_sensor_frame_affine.matrix().cast<float>();
    pcl::transformPointCloud(*input_ptr, *output_ptr, base_to_sensor_frame_matrix);
  }
  void callbackPointsCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_points_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_points_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *input_points_ptr);
    sensor_msgs::msg::PointCloud2 transform_cloud;

    geometry_msgs::msg::TransformStamped frame_transform =
      getTransform(map_frame_id_, msg->header.frame_id);
    double diff_x =
      frame_transform.transform.translation.x - prev_transform_.transform.translation.x;
    double diff_y =
      frame_transform.transform.translation.y - prev_transform_.transform.translation.y;
    const double distance = std::sqrt(std::pow(diff_x, 2) + std::pow(diff_y, 2));
    RCLCPP_DEBUG(get_logger(), "distance : %f", distance);
    if (distance < min_displacement_) return;
    prev_transform_ = frame_transform;

    transformPointCloud(input_points_ptr, output_points_ptr, frame_transform);

    saveMap(output_points_ptr, leaf_size_);
  }
  void saveMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud, double leaf_size)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter;
    voxel_grid_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid_filter.setInputCloud(map_cloud);
    voxel_grid_filter.filter(*voxel_grid_cloud);

    for(std::size_t i=0;i<voxel_grid_cloud->size();++i) {
      std::vector<int> nn_indices(1);
      std::vector<float> nn_dists(1);
      tree_->nearestKSearch(voxel_grid_cloud->points.at(i), 1, nn_indices, nn_dists);
      if(min_distance_threshold_ < std::sqrt(nn_dists.at(0))) {
        filtered_cloud->points.push_back(voxel_grid_cloud->points.at(i));
      }
    }

    *map_cloud_ += *filtered_cloud;
    tree_->setInputCloud(map_cloud_);

    pcl::io::savePCDFileBinary(save_path_ + "/map_concat.pcd", *map_cloud_);
    pcl::io::savePCDFileBinary(
      save_path_ + "/map_" + std::to_string(file_num_++) + ".pcd", *voxel_grid_cloud);
  }

private:
  std::string base_frame_id_;
  std::string map_frame_id_;
  std::string save_path_;
  std::string pcd_path_;

  double leaf_size_;
  double min_displacement_;
  double min_distance_threshold_;

  int file_num_{0};

  pcl::PointCloud<pcl::PointXYZ>::Ptr map_cloud_;
  pcl::search::Search<pcl::PointXYZ>::Ptr tree_;

  geometry_msgs::msg::TransformStamped prev_transform_;

  tf2_ros::Buffer tf2_buffer_{get_clock()};
  tf2_ros::TransformListener tf2_listener_{tf2_buffer_};

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr points_subscriber_;
};

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(ConcatSensorToMap)
