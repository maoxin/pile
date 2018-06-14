#include <iostream>
#include <vector>
#include <cmath>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <pcl/surface/mls.h>

// declare class
class Plan {
  public:
    int index;
    std::vector<float> normal;
    float z;
    Plan(int index, std::vector<float> normal, float z): index(index), normal(normal), z(z) {}
    // Plan(Plan &plan): index(plan.index), normal(plan.normal), z(plan.z) {} //some error here
};
class ByZ { 
  public:
    bool operator()(Plan const &a, Plan const &b) { 
        return a.z > b.z;
    }
};

// declare function
float median(std::vector<float> vec);
int plot(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
int plot(pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> &reg);
int remove_outliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
pcl::PointCloud <pcl::Normal>::Ptr compute_normals(const pcl::search::Search<pcl::PointXYZ>::Ptr &tree,
                                                   const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> compute_reg(const pcl::search::Search<pcl::PointXYZ>::Ptr &tree,
                                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                           const pcl::PointCloud <pcl::Normal>::Ptr &normals,
                                                           std::vector <pcl::PointIndices> &clusters);
float z_median(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               const std::vector<int> &indices);
std::vector<float> normal_median(const pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                 const std::vector<int> &indices);
int transform(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
              pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_transformed,
              const std::vector<float> &ground_normal);
int write_clusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                   const std::vector <pcl::PointIndices> &clusters,
                   std::string prefix);
int normalize_normal(std::vector<float> &normal);
std::vector<Plan> compute_plans(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                const std::vector <pcl::PointIndices> &clusters,
                                const pcl::PointCloud <pcl::Normal>::Ptr &normals);
bool is_two_ground(const std::vector<Plan> plans);
std::vector<float> compute_ground_normal(const std::vector<Plan> plans,
                                         bool two_ground_plan);
std::vector<Plan> update_plans(const std::vector<Plan> &old_plans,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_transformed,
                               const std::vector <pcl::PointIndices> &clusters,
                               const pcl::PointCloud <pcl::Normal>::Ptr &normals_transformed);
std::vector<int> compute_indices_outlier (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                          const std::vector <pcl::PointIndices> &clusters,
                                          const std::vector<int> cluster_indices);
std::vector<int> compute_indices_inlier (const std::vector <pcl::PointIndices> &clusters);
std::vector<int> compute_last_cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_transformed,
                                      const std::vector<float> &ground_normal_transformed,
                                      const pcl::PointCloud <pcl::Normal>::Ptr &normals_transformed,
                                      const std::vector<int> &outlier_indices,
                                      float ground_z);
int write_clusters_normals(const pcl::PointCloud <pcl::Normal>::Ptr &normals,
                           const std::vector <pcl::PointIndices> &clusters,
                           std::string prefix);
std::vector<int> compute_full_indices (const std::vector<int> perfect_cluster_indices,
                                       const std::vector<int> &last_cluster_indices);
std::vector<int> remove_outliers_from_last_cluster (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                    std::vector<int> &last_cluster_indices);
int smooth_cloud (pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud);

//-----------------------------------------------------------------------------------------------------//

// utinity
float median(std::vector<float> vec) {
  size_t size = vec.size();

  if (size == 0) {
    return 0;  // Undefined, really.
  }

  else {
    std::sort(vec.begin(), vec.end());
    if (size % 2 == 0) {
      return (vec[size / 2 - 1] + vec[size / 2]) / 2;
    }
    else {
      return vec[size / 2];
    }
  }
}

int plot(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::visualization::PCLVisualizer viewer ("viewer");
  viewer.setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler (cloud, 255, 255, 255);
  viewer.addPointCloud(cloud, cloud_color_handler, "cloud");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");
  viewer.addCoordinateSystem (1.0);
  // viewer.initCameraParameters ();

  while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
    viewer.spinOnce ();
  }
}

int plot(pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> &reg) {
  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::PCLVisualizer viewer ("Cluster viewer");
  viewer.setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(colored_cloud);
  viewer.addPointCloud<pcl::PointXYZRGB> (colored_cloud, rgb, "classes");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "classes");
  viewer.addCoordinateSystem (1.0);
  // viewer.initCameraParameters ();

  while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
    viewer.spinOnce ();
  }

  return 0;
}

int remove_outliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  // pcl::IndicesPtr indices (new std::vector <int>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (3.0);
  // sor.filter (*indices);
  sor.filter(*cloud);

  return 0;
}

pcl::PointCloud <pcl::Normal>::Ptr compute_normals(const pcl::search::Search<pcl::PointXYZ>::Ptr &tree,
                                                   const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);

  return normals;
}

pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> compute_reg(const pcl::search::Search<pcl::PointXYZ>::Ptr &tree,
                                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                           const pcl::PointCloud <pcl::Normal>::Ptr &normals,
                                                           std::vector <pcl::PointIndices> &clusters) {
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (1000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  // reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (1.5 / 180.0 * M_PI);
  // reg.setSmoothnessThreshold (180 / 180.0 * M_PI);
  reg.setCurvatureThreshold (10);

  reg.extract (clusters);

  return reg;
}

float z_median(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               const std::vector<int> &indices) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>(*cloud, indices));

  std::vector<float> vec_z;
  for (size_t i = 0; i < cloud_cluster->points.size(); i++) {
    vec_z.push_back(cloud_cluster->points[i].z);
  }

  return median(vec_z);
}

std::vector<float> normal_median(const pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                 const std::vector<int> &indices) {
  pcl::PointCloud<pcl::Normal>::Ptr normal_cluster (new pcl::PointCloud<pcl::Normal>(*normals, indices));

  std::vector<float> vec_x, vec_y, vec_z;
  for (size_t i = 0; i < normal_cluster->points.size(); i++) {
    vec_x.push_back(normal_cluster->points[i].normal_x);
    vec_y.push_back(normal_cluster->points[i].normal_y);
    vec_z.push_back(normal_cluster->points[i].normal_z);
  }

  std::vector<float> ground_normal;
  float m_x, m_y, m_z;
  m_x = median(vec_x);
  m_y = median(vec_y);
  m_z = median(vec_z);
  ground_normal.push_back(m_x);
  ground_normal.push_back(m_y);
  ground_normal.push_back(m_z);

  normalize_normal(ground_normal);

  return ground_normal;
}

int normalize_normal(std::vector<float> &normal) {
  float length = std::sqrt(std::pow(normal[0], 2) + std::pow(normal[1], 2) + std::pow(normal[2], 2));
  for (int i=0; i < normal.size(); i++) {
    normal[i] = normal[i] / length;
  }

  return 0;
}

int transform(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
              pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_transformed,
              const std::vector<float> &ground_normal) {
  float rotate_axis_x = ground_normal[1];
  float rotate_axis_y = -ground_normal[0];
  float rotate_axis_length = std::sqrt(std::pow(rotate_axis_x, 2) + std::pow(rotate_axis_y, 2));
  Eigen::Vector3f rotate_axis (rotate_axis_x / rotate_axis_length, rotate_axis_y / rotate_axis_length, 0.0);
  float rotate_angle = std::acos(ground_normal[2]);
  
  Eigen::Affine3f transform;
  transform = Eigen::AngleAxisf(rotate_angle, rotate_axis);

  pcl::transformPointCloud(*cloud, *cloud_transformed, transform);

  return 0;
}

int write_clusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                   const std::vector <pcl::PointIndices> &clusters,
                   std::string prefix="gridn_cloud") {
  for (int i=0; i < clusters.size(); i++) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>(*cloud, clusters[i].indices));
    pcl::io::savePCDFileASCII (prefix + std::to_string(i) + ".pcd", *cloud_cluster);
  }

  return 0;
}

int write_clusters_normals(const pcl::PointCloud <pcl::Normal>::Ptr &normals,
                           const std::vector <pcl::PointIndices> &clusters,
                           std::string prefix="gridn_normal") {
  for (int i=0; i < clusters.size(); i++) {
    pcl::PointCloud<pcl::Normal>::Ptr normal_cluster (new pcl::PointCloud<pcl::Normal>(*normals, clusters[i].indices));
    pcl::io::savePCDFileASCII (prefix + std::to_string(i) + ".pcd", *normal_cluster);
  }

  return 0;
}

std::vector<Plan> compute_plans(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                const std::vector <pcl::PointIndices> &clusters,
                                const pcl::PointCloud <pcl::Normal>::Ptr &normals) {
  std::vector<Plan> plans;
  for (int i=0; i < clusters.size(); i++) {
    float z = z_median(cloud, clusters[i].indices);
    std::vector<float> normal = normal_median(normals, clusters[i].indices);
    Plan plan (i, normal, z);
    plans.push_back(plan);
  }
  std::sort(plans.begin(), plans.end(), ByZ());

  return plans;
}

std::vector<Plan> update_plans(const std::vector<Plan> &old_plans,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_transformed,
                               const std::vector <pcl::PointIndices> &clusters,
                               const pcl::PointCloud <pcl::Normal>::Ptr &normals_transformed) {
  std::vector<Plan> plans;
  for (int i=0; i < old_plans.size(); i++) {
    float z = z_median(cloud_transformed, clusters[old_plans[i].index].indices);
    std::vector<float> normal = normal_median(normals_transformed, clusters[old_plans[i].index].indices);
    Plan plan (old_plans[i].index, normal, z);
    plan.z = z;
    plan.normal = normal;
    plans.push_back(plan);
  }

  return plans;
}

bool is_two_ground(const std::vector<Plan> plans) {
  bool two_ground_plan = false;
  if ((std::abs(plans[0].z - plans[1].z) < 1) &&
      (std::abs(std::acos(plans[0].normal[2]) - std::acos(plans[1].normal[2])) < M_PI * 2.0 / 180.0)) {
    two_ground_plan = true;
    std::cout << "Two clusters belong to ground" << std::endl;
  }

  return two_ground_plan;
}

std::vector<float> compute_ground_normal(const std::vector<Plan> plans,
                                         bool two_ground_plan) {
  std::vector<float> ground_normal;
  if (two_ground_plan) {
    float normal_x = (plans[0].normal[0] + plans[1].normal[0]) / 2;
    float normal_y = (plans[0].normal[1] + plans[1].normal[1]) / 2;
    float normal_z = (plans[0].normal[2] + plans[1].normal[2]) / 2;
    ground_normal.push_back(normal_x);
    ground_normal.push_back(normal_y);
    ground_normal.push_back(normal_z);

    normalize_normal(ground_normal);
  } else {
    float normal_x = plans[0].normal[0];
    float normal_y = plans[0].normal[1];
    float normal_z = plans[0].normal[2];
    ground_normal.push_back(normal_x);
    ground_normal.push_back(normal_y);
    ground_normal.push_back(normal_z);
  }

  return ground_normal;
}

std::vector<int> compute_indices_inlier (const std::vector <pcl::PointIndices> &clusters) {
  std::vector<int> cluster_indices(clusters[0].indices);
  for (int i = 1; i < clusters.size(); i++) {
    cluster_indices.insert(cluster_indices.end(), clusters[i].indices.begin(), clusters[i].indices.end());
  }
  std::sort(cluster_indices.begin(), cluster_indices.end());

  return cluster_indices;
}

std::vector<int> compute_full_indices (const std::vector<int> perfect_cluster_indices,
                                       const std::vector<int> &last_cluster_indices) {
  std::vector<int> cluster_indices(perfect_cluster_indices);
  cluster_indices.insert(cluster_indices.end(), last_cluster_indices.begin(), last_cluster_indices.end());

  return cluster_indices;
}

std::vector<int> compute_indices_outlier (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                          const std::vector <pcl::PointIndices> &clusters,
                                          const std::vector<int> cluster_indices) {
  std::vector<int> outlier_indices;
  int guard = 0;
  for (int i=0; i < cloud->size(); i++) {
    if (guard < cluster_indices.size()) {
      if (i < cluster_indices[guard]) {
        outlier_indices.push_back(i);
      } else {
        guard++;
      }
    } else {
      outlier_indices.push_back(i);
    }
  }

  return outlier_indices;
}

std::vector<int> remove_outliers_from_last_cluster (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                    std::vector<int> &last_cluster_indices) {

  std::vector<int> indices;
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (3.0);
  sor.filter(indices);


  std::sort(indices.begin(), indices.end());
  std::sort(last_cluster_indices.begin(), last_cluster_indices.end());

  std::vector<int> last_last_cluster_indices;
  int guard = 0;
  for (int i=0; i < last_cluster_indices.size(); i++) {
    if (guard < indices.size()) {
      if (last_cluster_indices[i] < indices[guard]) {}
      else {
        last_last_cluster_indices.push_back(last_cluster_indices[i]);
        guard++;
      }
    } else {
      break;
    }
  }

  return last_last_cluster_indices;
}

std::vector<int> compute_last_cluster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_transformed,
                                      const std::vector<float> &ground_normal_transformed,
                                      const pcl::PointCloud <pcl::Normal>::Ptr &normals_transformed,
                                      const std::vector<int> &outlier_indices,
                                      float ground_z) {
  float normal_x0, normal_y0, normal_z0;
  normal_x0 = ground_normal_transformed[0];
  normal_y0 = ground_normal_transformed[1];
  normal_z0 = ground_normal_transformed[2];

  std::vector<int> last_cluster;
  for (int i=0; i < outlier_indices.size(); i++) {
    float z = cloud_transformed->points[outlier_indices[i]].z;
    float curvature = normals_transformed->points[outlier_indices[i]].curvature;
    float normal_x, normal_y, normal_z;
    normal_x = normals_transformed->points[outlier_indices[i]].normal_x;
    normal_y = normals_transformed->points[outlier_indices[i]].normal_y;
    normal_z = normals_transformed->points[outlier_indices[i]].normal_z;

    float theta = std::acos(normal_x0 * normal_x + normal_y0 * normal_y + normal_z0 * normal_z) * 180.0 / M_PI;
    if ( (theta >= 0) && (theta <= 16) ) {
    // if (true) {
      if (z >= ground_z) {
        last_cluster.push_back(outlier_indices[i]);
      } else {}
    }
  }

  // last_cluster = remove_outliers_from_last_cluster (cloud_transformed, last_cluster);

  return last_cluster;
}

int smooth_cloud (pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud) {
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

  pcl::PointCloud<pcl::PointXYZ> mls_points;

  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
 
  // mls.setComputeNormals (true);

  mls.setInputCloud (cloud);
  mls.setPolynomialFit (true);
  mls.setSearchMethod (tree);
  mls.setSearchRadius (0.5);

  mls.process (mls_points);
  pcl::io::savePCDFile ("mls.pcd", mls_points);

  return 0;
}

//-----------------------------------------------------------------------------------------------------//

int main (int argc, char** argv) {
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[filenames[0]], *origin_cloud) == -1) {
    std::cout << "Cloud reading failed." << std::endl;
    return -1;
  }


  // filter the outliers and smooth the surface
  smooth_cloud(origin_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> ("mls.pcd", *cloud) == -1) {
    std::cout << "Cloud reading failed." << std::endl;
    return -1;
  }

  remove_outliers(cloud);

  // generate tree for search
  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ>> (new pcl::search::KdTree<pcl::PointXYZ>);

  // compute normals
  pcl::PointCloud <pcl::Normal>::Ptr normals = compute_normals(tree, cloud);

  // segmentation
  std::vector <pcl::PointIndices> clusters;
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg = compute_reg(tree, cloud, normals, clusters);

  // generate plan descriptor
  // detect the ground cluster (maybe there are two ground plan)
  std::vector<Plan> plans = compute_plans(cloud, clusters, normals);
  bool two_ground_plan = is_two_ground(plans);

  // get ground_normal
  std::vector<float> ground_normal = compute_ground_normal(plans, two_ground_plan);

  // transform
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed (new pcl::PointCloud<pcl::PointXYZ>);
  transform(cloud, cloud_transformed, ground_normal);
  pcl::PointCloud <pcl::Normal>::Ptr normals_transformed = compute_normals(tree, cloud_transformed);
  std::vector<Plan> plans_transformed = update_plans(plans, cloud_transformed, clusters, normals_transformed);

  // write_clusters_normals(normals_transformed, clusters, "gridn_normal_transformed");

  // get normal of ground after transformation
  std::vector<float> ground_normal_transformed = compute_ground_normal(plans_transformed, two_ground_plan);
  // std::cout << "ground normal x after trans: " << ground_normal_transformed[0] <<std::endl;
  // std::cout << "ground normal y after trans: " << ground_normal_transformed[1] <<std::endl;
  // std::cout << "ground normal z after trans: " << ground_normal_transformed[2] <<std::endl;

  // extract indices for outliers
  std::vector<int> cluster_indices = compute_indices_inlier(clusters);
  std::vector<int> outlier_indices = compute_indices_outlier(cloud, clusters, cluster_indices);
  float ground_z;
  if (two_ground_plan) {
    ground_z = (plans_transformed[0].z + plans_transformed[1].z) / 2;
  } else {
    ground_z = plans_transformed[0].z;
  }
  std::vector<int> last_cluster_indices = compute_last_cluster(cloud_transformed, ground_normal_transformed, normals_transformed, outlier_indices, ground_z);
  // pcl::PointCloud<pcl::Normal>::Ptr last_cluster (new pcl::PointCloud<pcl::Normal>(*normals_transformed, last_cluster_indices));
  // pcl::io::savePCDFileASCII ("last_cluster_normal.pcd", *last_cluster);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr last_cluster (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, last_cluster_indices));
  // pcl::io::savePCDFileASCII ("last_cluster_cloud.pcd", *last_cluster);

  // std::cout << "length of ouliers: " << outlier_indices.size() << std::endl;
  // std::cout << "length of last cluster: " << last_cluster_indices.size() << std::endl;

  // extract full indices
  std::vector<int> full_indices = compute_full_indices (cluster_indices, last_cluster_indices);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_full_indices (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, full_indices));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_stable_indices (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, cluster_indices));
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_last_indices (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, last_cluster_indices));



  // write if needed
  // write_clusters(cloud_transformed, clusters, "gridn_cloud_transformed_x");


  plot(cloud_full_indices);
  // plot(cloud_stable_indices);
  // plot(cloud_last_indices);
  // plot(cloud_transformed);
  // plot(cloud);
  // plot(reg);

  return 0;
}