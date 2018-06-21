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

#include <pcl/io/vtk_io.h>
#include <pcl/surface/gp3.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/surface/concave_hull.h>
#include <pcl/surface/convex_hull.h>

#include <fstream>
#include <iterator>

#include <algorithm>

// declare class
class Plan {
  public:
    int index;
    std::vector<float> normal;
    float x;
    float y;
    float z;
    float d2o;
    Plan(int index, std::vector<float> normal, float x, float y, float z): index(index), normal(normal), x(x), y(y), z(z) {
      d2o = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
    }
    // Plan(Plan &plan): index(plan.index), normal(plan.normal), z(plan.z) {} //some error here
};
class ByZ { 
  public:
    bool operator()(Plan const &a, Plan const &b) { 
        return a.z > b.z;
    }
};
class ByD2O {
  public:
    bool operator()(Plan const &a, Plan const &b) {
      return a.d2o < b.d2o;
    }
};

class GroundBoundray {
  public:
    float x;
    float y;
    float d;
    int index;
    GroundBoundray(float x, float y, float d, int index): x(x), y(y), d(d), index(index) {}
};
class ByD {
  public:
    bool operator()(GroundBoundray const &a, GroundBoundray const &b) {
      return a.d > b.d;
    }
};

class Circle {
  public:
    float cx;
    float cy;
    float r;
    Circle(float cx, float cy, float r): cx(cx), cy(cy), r(r) {}
};

class Point {
  public:
    float x;
    float y;
    Point(float x, float y): x(x), y(y) {}
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
// std::vector<Plan> update_plans(const std::vector<Plan> &old_plans,
//                                const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_transformed,
//                                const std::vector <pcl::PointIndices> &clusters,
//                                const pcl::PointCloud <pcl::Normal>::Ptr &normals_transformed);
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
std::vector<int> compute_indices_inlier_no_ground(const std::vector <pcl::PointIndices> &clusters,
                                                  const std::vector<Plan> &plans,
                                                  bool two_ground_plan);
std::vector<float> compute_center_least_squares(const pcl::PointCloud <pcl::PointXYZ>::Ptr &main_pile_cloud,
                                                const pcl::PointCloud <pcl::Normal>::Ptr &main_pile_normal);
Circle compute_center(const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud,
                      const std::vector<int> fornt_pile_indices);
float compute_r_main_pile(const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud_main_pile,
                          const Circle &ground_circle);
float compute_volume_4_cluster(const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud_transformed,
                               const std::vector<int> &clusters_indices,
                               float bottom_z);
float sign (Point &p1, Point &p2, Point &p3);
bool PointInTriangle (Point &pt, Point &v1, Point &v2, Point &v3);
pcl::PointCloud<pcl::PointXYZ>::Ptr compute_center_cloud_4_polygons(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                                                    const std::vector< pcl::Vertices > &polygons);
std::vector< pcl::Vertices> compute_lower_surface_polygons (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                                            const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud_polygon_center,
                                                            const std::vector< pcl::Vertices> &polygons,
                                                            const pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree);
float x_median(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               const std::vector<int> &indices);
float y_median(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               const std::vector<int> &indices);
bool is_two_front(const std::vector<Plan> plans);

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

float x_median(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               const std::vector<int> &indices) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>(*cloud, indices));

  std::vector<float> vec_x;
  for (size_t i = 0; i < cloud_cluster->points.size(); i++) {
    vec_x.push_back(cloud_cluster->points[i].x);
  }

  return median(vec_x);
}

float y_median(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               const std::vector<int> &indices) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>(*cloud, indices));

  std::vector<float> vec_y;
  for (size_t i = 0; i < cloud_cluster->points.size(); i++) {
    vec_y.push_back(cloud_cluster->points[i].y);
  }

  return median(vec_y);
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
    float x = x_median(cloud, clusters[i].indices);
    float y = y_median(cloud, clusters[i].indices);
    float z = z_median(cloud, clusters[i].indices);
    std::vector<float> normal = normal_median(normals, clusters[i].indices);
    Plan plan (i, normal, x, y, z);
    plans.push_back(plan);
  }
  std::sort(plans.begin(), plans.end(), ByD2O());

  return plans;
}

// std::vector<Plan> update_plans(const std::vector<Plan> &old_plans,
//                                const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_transformed,
//                                const std::vector <pcl::PointIndices> &clusters,
//                                const pcl::PointCloud <pcl::Normal>::Ptr &normals_transformed) {
//   std::vector<Plan> plans;
//   for (int i=0; i < old_plans.size(); i++) {
//     float z = z_median(cloud_transformed, clusters[old_plans[i].index].indices);
//     std::vector<float> normal = normal_median(normals_transformed, clusters[old_plans[i].index].indices);
//     Plan plan (old_plans[i].index, normal, x, y, z);
//     plan.z = z;
//     plan.normal = normal;
//     plans.push_back(plan);
//   }

//   return plans;
// }

bool is_two_ground(const std::vector<Plan> plans) {
  bool two_ground_plan = false;
  if ((std::abs(plans[0].z - plans[1].z) < 1) &&
      (std::abs(std::acos(plans[0].normal[2]) - std::acos(plans[1].normal[2])) < M_PI * 2.0 / 180.0)) {
    two_ground_plan = true;
    std::cout << "Two clusters belong to ground" << std::endl;
  }

  return two_ground_plan;
}

bool is_two_front(const std::vector<Plan> plans) {
  bool two_front = false;

  std::cout << "diff of distance to O of the first two plans" << std::abs(plans[0].d2o - plans[1].d2o) << std::endl;
  if (std::abs(plans[0].d2o - plans[1].d2o) < 5) {
    two_front = true;
    std::cout << "Two clusters belong to front pile" << std::endl;
  }

  return two_front;
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

std::vector<int> compute_indices_inlier_no_ground(const std::vector <pcl::PointIndices> &clusters,
                                                  const std::vector<Plan> &plans,
                                                  bool two_ground_plan) {
  int start;
  if (two_ground_plan) {
    start = 2;
  } else {
    start = 1;
  }

  std::vector<int> cluster_indices(clusters[plans[start].index].indices);
  for (int i = start+1; i < plans.size(); i++) {
    cluster_indices.insert(cluster_indices.end(), clusters[plans[i].index].indices.begin(),
                           clusters[plans[i].index].indices.end());
  }
  // std::sort(cluster_indices.begin(), cluster_indices.end());

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

std::vector<float> compute_center_least_squares(const pcl::PointCloud <pcl::PointXYZ>::Ptr &main_pile_cloud,
                                                const pcl::PointCloud <pcl::Normal>::Ptr &main_pile_normal) {
  // use least squares to calculate the center point
  float a = 0;
  float b = 0;
  float c = 0;
  float d = 0;
  float e = 0;
  float f = 0;
  float g = 0;

  for (int i=0; i < main_pile_cloud->points.size(); i++) {
    a += std::pow(main_pile_normal->points[i].normal_x, 2);
    b += std::pow(main_pile_normal->points[i].normal_y, 2);
    c += main_pile_normal->points[i].normal_x * main_pile_normal->points[i].normal_y;
    d += std::pow(main_pile_normal->points[i].normal_y, 2) * main_pile_cloud->points[i].x;
    e += std::pow(main_pile_normal->points[i].normal_x, 2) * main_pile_cloud->points[i].y;
    f += main_pile_normal->points[i].normal_x * main_pile_normal->points[i].normal_y * main_pile_cloud->points[i].y;
    f += main_pile_normal->points[i].normal_x * main_pile_normal->points[i].normal_y * main_pile_cloud->points[i].x;
  }

  float center_x = ((d - f) * c - (g - e) * b) / (a * b - std::pow(c, 2));
  float center_y = ((d - f) * a - (g - e) * c) / (a * b - std::pow(c, 2));

  std::vector<float> center;
  center.push_back(center_x);
  center.push_back(center_y);

  return center;
}

Circle compute_center(const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud,
                      const std::vector<int> fornt_pile_indices) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_front (new pcl::PointCloud<pcl::PointXYZ>(*cloud, fornt_pile_indices));
  for (int i=0; i < cloud_front->size(); i++) {
    cloud_front->points[i].z = 0;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ConcaveHull<pcl::PointXYZ> chull;
  chull.setInputCloud(cloud_front);
  chull.setAlpha(3.0);
  chull.setDimension(2);
  chull.reconstruct (*cloud_hull);

  std::vector<GroundBoundray> gbs;
  for (int i=0; i < cloud_hull->size(); i++) {
    float x = cloud_hull->points[i].x;
    float y = cloud_hull->points[i].y;
    float d = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
    GroundBoundray gb(x, y, d, i);
    gbs.push_back(gb);
  }
  std::sort(gbs.begin(), gbs.end(), ByD());

  std::vector<int> gb_idx;
  for (int i=0; i < gbs.size()/2; i++) {
    gb_idx.push_back(gbs[i].index);
  }

  // plot if needed
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull_v (new pcl::PointCloud<pcl::PointXYZ>(*cloud_hull));
  // (*cloud_hull_v).insert( (*cloud_hull_v).end(), 1, pcl::PointXYZ(2.12665, 29.5835, 0) );
  // plot(cloud_hull_v);

  float center_x, center_y;

  float x1, y1, x2, y2, x3, y3;
  x1 = cloud_hull->points[gb_idx[0]].x;
  y1 = cloud_hull->points[gb_idx[0]].y;
  x2 = cloud_hull->points[gb_idx[gb_idx.size()/2]].x;
  y2 = cloud_hull->points[gb_idx[gb_idx.size()/2]].y;
  x3 = cloud_hull->points[gb_idx.size()-1].x;
  y3 = cloud_hull->points[gb_idx.size()-1].y;

  float mu_x_12 = (x1 + x2) / 2;
  float mu_y_12 = (y1 + y2) / 2;
  float mu_x_23 = (x2 + x3) / 2;
  float mu_y_23 = (y2 + y3) / 2;

  float k_12, k_23;
  if (x1 - x2 == 0) {
    center_y = mu_y_12;
    k_23 = (y3 - y2) / (x3 - x2);
    center_x = -center_y * k_23 + mu_x_23 + mu_y_23 * k_23;
  } else if (x3 - x2 == 0) {
    center_y = mu_y_23;
    k_12 = (y2 - y1) / (x2 - x1);
    center_x = -center_y * k_12 + mu_x_12 + mu_y_12 * k_12;
  } else {
    k_12 = (y1 - y2) / (x1 - x2);
    k_23 = (y2 - y3) / (x2 - x3);
    
    if (k_12 == 0) {
      center_x = mu_x_12;
      center_y = (mu_x_12 - mu_x_23 + mu_y_12 * k_12 - mu_y_23 * k_23) / (k_12 - k_23);
    } else if (k_23 == 0) {
      center_x = mu_x_23;
      center_y = (mu_x_12 - mu_x_23 + mu_y_12 * k_12 - mu_y_23 * k_23) / (k_12 - k_23);
    } else {
      center_y = (mu_x_12 - mu_x_23 + mu_y_12 * k_12 - mu_y_23 * k_23) / (k_12 - k_23);
      center_x = (mu_y_12 - mu_y_23 + mu_x_12 / k_12 - mu_x_23 / k_23) / (1 / k_12 - 1/ k_23);
    }
  }

  float r1, r2, r3;
  r1 = std::sqrt(std::pow(x1 - center_x, 2) + std::pow(y1 - center_y, 2));
  r2 = std::sqrt(std::pow(x2 - center_x, 2) + std::pow(y2 - center_y, 2));
  r3 = std::sqrt(std::pow(x3 - center_x, 2) + std::pow(y3 - center_y, 2));

  float r = (r1 + r2 + r3) / 3;

  Circle front_circle(center_x, center_y, r);

  return front_circle;
}

float compute_r_main_pile(const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud_main_pile,
                          const Circle &ground_circle) {
  float max_r = 0;
  float cx = ground_circle.cx;
  float cy = ground_circle.cy;
  for (int i=0; i < cloud_main_pile->size(); i++) {
    float x = cloud_main_pile->points[i].x;
    float y = cloud_main_pile->points[i].y;
    
    float r = std::sqrt(std::pow(x-cx, 2) + std::pow(y-cy, 2));
    if (r > max_r) {
      max_r = r;
    }
  }

  return max_r;
}

float sign (Point &p1, Point &p2, Point &p3)
{
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool PointInTriangle (Point &pt, Point &v1, Point &v2, Point &v3)
{
    bool b1, b2, b3;

    b1 = sign(pt, v1, v2) < 0.0f;
    b2 = sign(pt, v2, v3) < 0.0f;
    b3 = sign(pt, v3, v1) < 0.0f;

    return ((b1 == b2) && (b2 == b3));
}

pcl::PointCloud<pcl::PointXYZ>::Ptr compute_center_cloud_4_polygons(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                                                    const std::vector< pcl::Vertices > &polygons) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_polygon_center (new pcl::PointCloud<pcl::PointXYZ>);
  cloud_polygon_center->width = polygons.size();
  cloud_polygon_center->height = 1;
  cloud_polygon_center->points.resize (cloud_polygon_center->width * cloud_polygon_center->height);

  for (int i=0; i < polygons.size(); i++) {
    float cx = 0;
    float cy = 0;
    for (int j=0; j < polygons[i].vertices.size(); j++) {
      cx += cloud_hull->points[polygons[i].vertices[j]].x;
      cy += cloud_hull->points[polygons[i].vertices[j]].y;
    }
    cx /= polygons[i].vertices.size();
    cy /= polygons[i].vertices.size();

    cloud_polygon_center->points[i].x = cx;
    cloud_polygon_center->points[i].y = cy;
    cloud_polygon_center->points[i].z = 0;
  }

  return cloud_polygon_center;
}

std::vector< pcl::Vertices> compute_lower_surface_polygons (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                                            const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud_polygon_center,
                                                            const std::vector< pcl::Vertices> &polygons,
                                                            const pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree) {

  int K = 1500;
  // float radius = 1;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);

  std::vector< pcl::Vertices> polygons2use;
  for (int i=0; i < polygons.size(); i++) {
    pcl::PointXYZ searchPoint = cloud_polygon_center->points[i];
    Point pt = Point(cloud_polygon_center->points[i].x, cloud_polygon_center->points[i].y);

    float min_z = (cloud_hull->points[polygons[i].vertices[0]].z +
                   cloud_hull->points[polygons[i].vertices[1]].z +
                   cloud_hull->points[polygons[i].vertices[2]].z) / 3;
    bool remain = true;

    if ( kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) {
    // if ( kdtree.radiusSearch(searchPoint, radius, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) {
      for (size_t j = 0; j < pointIdxNKNSearch.size (); j++) {
        Point v1 = Point(cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[0]].x,
                         cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[0]].y);
        Point v2 = Point(cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[1]].x,
                         cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[1]].y);
        Point v3 = Point(cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[2]].x,
                         cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[2]].y);
        if (PointInTriangle(pt, v1, v2, v3)) {
          float polygon_z = (cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[0]].z +
                             cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[1]].z +
                             cloud_hull->points[polygons[pointIdxNKNSearch[j]].vertices[2]].z ) / 3;
          if (polygon_z < min_z) {
            remain = false;
            break;
          }
        }
      }
    }

    if (remain) {
      polygons2use.push_back(polygons[i]);
    }
  }

  // std::cout << "origin size: " << polygons.size() << std::endl;
  // std::cout << "current size: " << polygons2use.size() << std::endl;
  return polygons2use;
}

float compute_volume_4_cluster(const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud_transformed,
                               const std::vector<int> &clusters_indices,
                               float bottom_z) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, clusters_indices));

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
  std::vector< pcl::Vertices > polygons;

  pcl::ConcaveHull<pcl::PointXYZ> chull;
  chull.setInputCloud(cloud_cluster);
  chull.setAlpha(3.0);
  // pcl::io::saveVTKFile ("mesh_full_chull.3.0.vtk", triangles);
  chull.reconstruct (*cloud_hull, polygons);

  // --remove some polygons
  std::vector< pcl::Vertices> lower_surface_polygons;
  if (polygons[0].vertices.size() == 3) {
    // 1. construct center point cloud for polygons
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_polygon_center = compute_center_cloud_4_polygons(cloud_hull, polygons);

    // 2. just remain those in the lower surface (use kdtree to fast the process)
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_polygon_center);

    lower_surface_polygons = compute_lower_surface_polygons (cloud_hull, cloud_polygon_center, polygons, kdtree);
  } else {
    lower_surface_polygons = polygons;
  }

  // --compute volume under the lower polygons

  float volume_cluster = 0;
  for (int i=0; i < lower_surface_polygons.size(); i++) {
    std::vector<uint32_t> indices = lower_surface_polygons[i].vertices;

    // using signed area to calculate the area of the project
    float area_project = 0;
    for (int j=0; j < indices.size(); j++) {
      float x0 = cloud_hull->points[indices[j]].x;
      float y0 = cloud_hull->points[indices[j]].y;
      float x1 = cloud_hull->points[indices[(j+1)%indices.size()]].x;
      float y1 = cloud_hull->points[indices[(j+1)%indices.size()]].y;
      area_project += x0 * y1 - x1 * y0;
    }
    area_project = std::abs(area_project) / 2;

    // calcualte the centroid z
    float cz = 0;
    if (indices.size() == 3) {
      for (int j=0; j < indices.size(); j++) {
        cz += cloud_hull->points[indices[j]].z;
      }
      cz /= 3;
    } else {
      // https://math.stackexchange.com/questions/1338/compute-the-centroid-of-a-3d-planar-polygon-without-projecting-it-to-specific-pl
      // normals
      float e1x = cloud_hull->points[indices[0]].x - cloud_hull->points[indices[indices.size()/4]].x;
      float e1y = cloud_hull->points[indices[0]].y - cloud_hull->points[indices[indices.size()/4]].y;
      float e1z = cloud_hull->points[indices[0]].z - cloud_hull->points[indices[indices.size()/4]].z;
      float e2x = cloud_hull->points[indices[0]].x - cloud_hull->points[indices[indices.size()* 3 /4]].x;
      float e2y = cloud_hull->points[indices[0]].y - cloud_hull->points[indices[indices.size()* 3 /4]].y;
      float e2z = cloud_hull->points[indices[0]].z - cloud_hull->points[indices[indices.size()* 3 /4]].z;

      float nx = e1y * e2z - e1z * e2y;
      float ny = e1z * e2x - e1x * e2z;
      float nz = e1x * e2y - e1y * e2x;

      e2x = e1y * nz - e1z * ny;
      e2y = e1z * nx - e1x * nz;
      e2z = e1x * ny - e1y * nx;

      float length_e1 = std::sqrt(std::pow(e1x, 2) + std::pow(e1y, 2) + std::pow(e1z, 2));
      e1x /= length_e1;
      e1y /= length_e1;
      e1z /= length_e1;
      float length_e2 = std::sqrt(std::pow(e2x, 2) + std::pow(e2y, 2) + std::pow(e2z, 2));
      e2x /= length_e2;
      e2y /= length_e2;
      e2z /= length_e2;

      // anchor point
      float x_a = cloud_hull->points[indices[0]].x;
      float y_a = cloud_hull->points[indices[0]].y;
      float z_a = cloud_hull->points[indices[0]].z;

      float area_signed = 0;
      float Cx = 0;
      float Cy = 0;
      for (int j=0; j < indices.size(); j++) {
        float x0 = cloud_hull->points[indices[j]].x;
        float y0 = cloud_hull->points[indices[j]].y;
        float z0 = cloud_hull->points[indices[j]].z;
        float x1 = cloud_hull->points[indices[(j+1)%indices.size()]].x;
        float y1 = cloud_hull->points[indices[(j+1)%indices.size()]].y;
        float z1 = cloud_hull->points[indices[(j+1)%indices.size()]].z;

        // transform
        float t_x0 = (x0 - x_a) * e1x + (y0 - y_a) * e1y + (z0 - z_a) * e1z;
        float t_y0 = (x0 - x_a) * e2x + (y0 - y_a) * e2y + (z0 - z_a) * e2z;
        float t_x1 = (x1 - x_a) * e1x + (y1 - y_a) * e1y + (z1 - z_a) * e1z;
        float t_y1 = (x1 - x_a) * e2x + (y1 - y_a) * e2y + (z1 - z_a) * e2z;

        // compute singed area
        area_signed += t_x0 * t_y1 - t_x1 * t_y0;
        Cx += (t_x0 + t_x1) * (t_x0 * t_y1 - t_x1 * t_y0);
        Cy += (t_y0 + t_y1) * (t_x0 * t_y1 - t_x1 * t_y0);
      }

      // center point after transformed
      area_signed /= 2;
      Cx /= (6 * area_signed);
      Cy /= (6 * area_signed);
      
      // center point in origin coordinate
      cz = z_a + e1z * Cx + e2z * Cy;
    }

    if (cz - bottom_z > 0) {
      volume_cluster += area_project * (cz - bottom_z);
      // volume_cluster += area_project;
    } else {}
  }

  // std::cout << "polygon num: " << lower_surface_polygons.size() << std::endl;
  // std::cout << "area sum: " << volume_cluster << std::endl;
  return volume_cluster;
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

  // std::vector<float> z;
  // for (int i=0; i < cloud->size(); i++) {
  //   z.push_back(cloud->points[i].z);
  // }

  // float min_z = *std::min_element(z.begin(), z.end());
  // std::cout << min_z << std::endl;
  // std::vector<int> idx;
  // int s = cloud->size();
  // for (int i=0; i < s; i++) {
  //   (*cloud).insert( (*cloud).end(), 1, pcl::PointXYZ(cloud->points[i].x, cloud->points[i].y, min_z) );
  // }

  // plot(cloud);
  

  // generate tree for search
  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ>> (new pcl::search::KdTree<pcl::PointXYZ>);

  // compute normals
  pcl::PointCloud <pcl::Normal>::Ptr normals = compute_normals(tree, cloud);

  // std::vector<int> idx;
  // for (int i=0; i < normals->size(); i++) {
  //   if (std::abs(normals->points[i].normal_z) > 0.8) {
  //     idx.push_back(i);
  //   }
  // }

  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>(*cloud, idx));
  // plot(cloud1);

  // segmentation
  std::vector <pcl::PointIndices> clusters;
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg = compute_reg(tree, cloud, normals, clusters);

  // generate plan descriptor
  // detect the front pile cluster (maybe there are two front pile plan)
  std::vector<Plan> plans = compute_plans(cloud, clusters, normals);
  // bool two_ground_plan = is_two_ground(plans);
  bool two_front = is_two_front(plans);
  std::vector<int> front_pile_indices(clusters[plans[0].index].indices);
  // if (two_front) {
  //   front_pile_indices.insert(front_pile_indices.end(), clusters[plans[1].index].indices.begin(), clusters[plans[1].index].indices.end());
  // }

  // -- compute the center of the pile

  Circle front_circle = compute_center(cloud, front_pile_indices);
  std::cout << "center x: " << front_circle.cx << std::endl;
  std::cout << "center y: " << front_circle.cy << std::endl;
  std::cout << "ground circle r: " << front_circle.r << std::endl;

  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 (new pcl::PointCloud<pcl::PointXYZ>(*cloud, front_pile_indices));

  // for (int i=0; i < cloud->size(); i++) {
    // cloud->points[i].z = -5;
  // }
  // (*cloud).insert( (*cloud).end(), 1, pcl::PointXYZ(front_circle.cx, front_circle.cy, 0) );
  // plot(cloud);


  // get ground_normal
  // std::vector<float> ground_normal = compute_ground_normal(plans, two_ground_plan);

  // // transform
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_transformed (new pcl::PointCloud<pcl::PointXYZ>);
  // transform(cloud, cloud_transformed, ground_normal);
  // pcl::PointCloud <pcl::Normal>::Ptr normals_transformed = compute_normals(tree, cloud_transformed);
  // std::vector<Plan> plans_transformed = update_plans(plans, cloud_transformed, clusters, normals_transformed);

  // // write_clusters_normals(normals_transformed, clusters, "gridn_normal_transformed");

  // // get normal of ground after transformation
  // std::vector<float> ground_normal_transformed = compute_ground_normal(plans_transformed, two_ground_plan);

  // // detect the main pile cluster
  // float angle_repose = 0.0;
  // int main_pile_index_in_plans = 0;
  // for (int i=0; i < plans_transformed.size(); i++) {
  //   float angle = std::acos(plans_transformed[i].normal[0] * ground_normal_transformed[0] +
  //                           plans_transformed[i].normal[1] * ground_normal_transformed[1] +
  //                           plans_transformed[i].normal[2] * ground_normal_transformed[2]) * 180.0 / M_PI;
  //   if (angle > angle_repose) {
  //     angle_repose = angle;
  //     main_pile_index_in_plans = i;
  //   }
  // }
  // std::cout << "angle pose: " << angle_repose << std::endl;

  // // extract indices for outliers
  // std::vector<int> cluster_indices = compute_indices_inlier(clusters);
  // std::vector<int> outlier_indices = compute_indices_outlier(cloud, clusters, cluster_indices);
  // float ground_z;
  // if (two_ground_plan) {
  //   ground_z = (plans_transformed[0].z + plans_transformed[1].z) / 2;
  // } else {
  //   ground_z = plans_transformed[0].z;
  // }
  // std::vector<int> last_cluster_indices = compute_last_cluster(cloud_transformed, ground_normal_transformed, normals_transformed, outlier_indices, ground_z);

  // // -- extract indices and generate point clouds from different aspects
  // std::vector<int> full_indices = compute_full_indices (cluster_indices, last_cluster_indices);
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_full_indices (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, full_indices));


  // // std::vector<int> cluster_indices_no_ground = compute_indices_inlier_no_ground(clusters, plans_transformed, two_ground_plan);
  // // std::vector<int> full_no_ground_indices = compute_full_indices (cluster_indices_no_ground, last_cluster_indices);
  // // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_full_no_ground_indices (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, full_no_ground_indices));

  // // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_stable_indices (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, cluster_indices));
  // // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_stable_no_ground_indices (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, cluster_indices_no_ground));

  // // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_last_indices (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, last_cluster_indices));

  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_main_pile (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, clusters[plans_transformed[main_pile_index_in_plans].index].indices));
  // // pcl::PointCloud<pcl::Normal>::Ptr normal_main_pile (new pcl::PointCloud<pcl::Normal>(*normals_transformed, clusters[plans_transformed[main_pile_index_in_plans].index].indices));

  // std::vector<int> ground_indices(clusters[plans_transformed[0].index].indices);
  // if (two_ground_plan) {
  //   ground_indices.insert(ground_indices.end(), clusters[plans_transformed[1].index].indices.begin(), clusters[plans_transformed[1].index].indices.end());
  // }
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground (new pcl::PointCloud<pcl::PointXYZ>(*cloud_transformed, ground_indices));
  

  // // -- construct triangle for the full dataset
  // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
  // // std::vector< pcl::Vertices > polygons;
  // // pcl::ConcaveHull<pcl::PointXYZ> chull;
  // pcl::ConcaveHull<pcl::PointXYZ> chull;

  // chull.setInputCloud(cloud_full_indices);
  // chull.setAlpha(3.0);
  // chull.setKeepInformation(true);
  // // chull.setDimension(2);
  // pcl::PolygonMesh triangles;
  // chull.reconstruct(triangles);
  // pcl::io::saveVTKFile ("mesh_full_chull.3.0.vtk", triangles);
  // chull.reconstruct (*cloud_hull);

  // // pcl::PointCloud <pcl::Normal>::Ptr normal_hull =  compute_normals(tree, cloud_hull);
  // // pcl::io::savePCDFileASCII ("normal_hull.pcd", *normal_hull);

  // // -- compute volume
  // // 1. generate vector of indices for clusters to be calculated
  // std::vector< std::vector<int> > clusters4volume;
  
  // int start = 1;
  // if (two_ground_plan) {
  //   start = 2;
  // }

  // for (int i = start; i < plans_transformed.size(); i++) {
  //   clusters4volume.push_back(clusters[plans_transformed[i].index].indices);
  // }
  // clusters4volume.push_back(last_cluster_indices);

  // // 2. use minimum z of main pile as the the base for calculating volume
  // std::vector<float> main_pile_zs;
  // for (int i=0; i < cloud_main_pile->size(); i++) {
  //   main_pile_zs.push_back(cloud_main_pile->points[i].z);
  // }
  // float bottom_z = *std::min_element(main_pile_zs.begin(), main_pile_zs.end());

  // // 3. compute the volume
  // float volume = 0;
  // for (int i=0; i < clusters4volume.size(); i++) {
  //   // std::cout << "cluster: " << i << std::endl;
  //   volume += compute_volume_4_cluster(cloud_transformed,
  //                                      clusters4volume[i],
  //                                      bottom_z);
  // }

  // // -- compute the radius for making up the missing volume
  // Circle ground_circle = compute_center(cloud_transformed, ground_indices);
  // // std::cout << "center x: " << ground_circle.cx << std::endl;
  // // std::cout << "center y: " << ground_circle.cy << std::endl;
  // std::cout << "ground circle r: " << ground_circle.r << std::endl;

  // float r_ridge =  compute_r_main_pile(cloud_main_pile, ground_circle);
  // float width_cross_section_main_pile = r_ridge - ground_circle.r;
  // std::cout << "width of cross section of pile: " << width_cross_section_main_pile << std::endl;

  // float volume_back = ((2 * ground_circle.r + 3 * width_cross_section_main_pile) /
  //                      (2 * ground_circle.r + width_cross_section_main_pile) *
  //                       volume);
  // float volume_with_back = volume + volume_back;
  // std::cout << "volum form point cloud: " << volume << std::endl;
  // std::cout << "volum back: " << volume_back << std::endl;
  // std::cout << "volum with back: " << volume_with_back << std::endl;
  // ofstream result_file;
  // result_file.open ("result.csv");
  // result_file << "angle_pose,volum_from_point_cloud,volume_back,volume_with_back\n";
  // result_file << (std::to_string(angle_repose) + "," + std::to_string(volume) +
  //                 "," + std::to_string(volume_back) + "," +
  //                 std::to_string(volume_with_back) + "\n");
  // result_file.close();

  // // -- interactive 3-D plot
  // // plot(cloud_hull);
  // // 1. for plot the center
  // // (*cloud_main_pile).insert( (*cloud_main_pile).end(), 1, pcl::PointXYZ(ground_circle.cx, ground_circle.cy, ground_z) );
  // // plot(cloud_main_pile);

  // // 2. plot the main pile
  // // plot(cloud_main_pile);

  // // 3. other
  // // plot(cloud_full_indices);
  // // plot(cloud_stable_indices);
  // // plot(cloud_last_indices);
  // // plot(cloud_transformed);
  // // plot(cloud);
  // // plot(reg);

  return 0;
}