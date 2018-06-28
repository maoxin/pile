#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <functional>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/io/vtk_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/concave_hull.h>

// declare class
class EdgeInfo {
  public:
    float cx;
    float cy;
    std::vector<int> edge_idx;
    EdgeInfo(float cx, float cy, const std::vector<int> edge_idx): cx(cx), cy(cy), edge_idx(edge_idx) {}
};

class Result {
  public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final;
    float volume;
    
    Result(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_final, float volume): cloud_final(cloud_final), volume(volume) {}
};
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

class Point {
  public:
    float x;
    float y;
    Point(float x, float y): x(x), y(y) {}
};

class PointPotentialEdge {
  public:
    int idx;
    float rho;
    float z;
    float z2base;
    PointPotentialEdge(int idx, float rho, float z): idx(idx), rho(rho), z(z), z2base(0.0) {}
};

class ByRhoPPE {
  public:
    bool operator()(PointPotentialEdge const &a, PointPotentialEdge const &b) {
      return a.rho < b.rho;
    }
};

class PolygonRaster {
  public:
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_polygon_raster;
    std::vector<int> polygon_raster_idx;
    PolygonRaster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_polygon_raster, const std::vector<int> &polygon_raster_idx):
      cloud_polygon_raster(cloud_polygon_raster), polygon_raster_idx(polygon_raster_idx) {}
};

// declare function
float median(std::vector<float> vec);
int plot(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
int plot(pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> &reg);
pcl::PointCloud<pcl::PointXYZ>::Ptr remove_outliers(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                    float bottom_z, float top_z);
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
int normalize_normal(std::vector<float> &normal);
std::vector<Plan> compute_plans(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                const std::vector <pcl::PointIndices> &clusters,
                                const pcl::PointCloud <pcl::Normal>::Ptr &normals);
int smooth_cloud (pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud);
float sign (Point &p1, Point &p2, Point &p3);
bool PointInTriangle (Point &pt, Point &v1, Point &v2, Point &v3);

float x_median(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               const std::vector<int> &indices);
float y_median(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
               const std::vector<int> &indices);
bool is_two_front(const std::vector<Plan> plans);
pcl::PointCloud<pcl::PointXYZ>::Ptr transform2cylinder(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                       float cx, float cy);
pcl::PointCloud<pcl::PointXYZ>::Ptr remove_wall(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
pcl::PointCloud<pcl::PointXYZ>::Ptr complete_back_data(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                       const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_cylinder,
                                                       const pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                                       const std::vector<int> &back_indices,
                                                       const std::vector<int> &edge_idx,
                                                       float bottom_z, float cx, float cy);

pcl::PointCloud<pcl::PointXYZ>::Ptr refine_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                 float bottom_z,
                                                 bool ground);
pcl::PointCloud<pcl::PointXYZ>::Ptr compute_upper_surface_cloud (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_refine,
                                                                 const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                                                 const PolygonRaster &polygon_raster,
                                                                 const std::vector< pcl::Vertices> &polygons,
                                                                 float bottom_z,
                                                                 float raster_size);
Result compute_volume(const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud_refine,
                      float bottom_z, float raster_size);
PolygonRaster compute_polygon_raster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                     const std::vector< pcl::Vertices > &polygons,
                                     float raster_size);
std::vector<int> compute_edge(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_cylinder);
EdgeInfo auto_compute_edge(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);
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

  return 0;
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

pcl::PointCloud<pcl::PointXYZ>::Ptr remove_outliers(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                    float bottom_z) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (200);
  sor.setStddevMulThresh (2.0);
  sor.filter(*cloud_filtered);

  return cloud_filtered;
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

  std::vector<float> m_normal;
  float m_x, m_y, m_z;
  m_x = median(vec_x);
  m_y = median(vec_y);
  m_z = median(vec_z);
  m_normal.push_back(m_x);
  m_normal.push_back(m_y);
  m_normal.push_back(m_z);

  normalize_normal(m_normal);

  return m_normal;
}

int normalize_normal(std::vector<float> &normal) {
  float length = std::sqrt(std::pow(normal[0], 2) + std::pow(normal[1], 2) + std::pow(normal[2], 2));
  for (int i=0; i < normal.size(); i++) {
    normal[i] = normal[i] / length;
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

bool is_two_front(const std::vector<Plan> plans) {
  bool two_front = false;

  std::cout << "diff of distance to O of the first two plans: " << std::abs(plans[0].d2o - plans[1].d2o) << std::endl;
  if (std::abs(plans[0].d2o - plans[1].d2o) < 5) {
    two_front = true;
    std::cout << "Two clusters belong to front pile" << std::endl;
  }

  return two_front;
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

pcl::PointCloud<pcl::PointXYZ>::Ptr compute_upper_surface_cloud (const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_refine,
                                                                 const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                                                 const PolygonRaster &polygon_raster,
                                                                 const std::vector< pcl::Vertices> &polygons,
                                                                 float bottom_z,
                                                                 float raster_size=0.2) {
  std::vector<float> vec_x, vec_y;
  for (int i=0; i < cloud_refine->size(); i++) {
    vec_x.push_back(cloud_refine->points[i].x);
    vec_y.push_back(cloud_refine->points[i].y);
  }

  float max_x = *std::max_element(vec_x.begin(), vec_x.end());
  float min_x = *std::min_element(vec_x.begin(), vec_x.end());
  float max_y = *std::max_element(vec_y.begin(), vec_y.end());
  float min_y = *std::min_element(vec_y.begin(), vec_y.end());
  
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(polygon_raster.cloud_polygon_raster);
  // int K = 1500;
  // std::vector<int> pointIdxNKNSearch(K);
  // std::vector<float> pointNKNSquaredDistance(K);
  float radius = raster_size;

  std::vector< pcl::Vertices> polygons2use;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final (new pcl::PointCloud<pcl::PointXYZ>);
  
  // for (int i=0; i < polygons.size(); i++) {
  for (float i=min_x; i <= max_x; i+=raster_size) {
    for (float j=min_y; j <= max_y; j+=raster_size) {
      std::vector<int> pointIdxNKNSearch;
      std::vector<float> pointNKNSquaredDistance;
      
      pcl::PointXYZ searchPoint;
      searchPoint.x = i;
      searchPoint.y = j;
      searchPoint.z = 0;
      Point pt = Point(searchPoint.x, searchPoint.y);

      float max_z = -999;
      int remain_idx = -1;

      // if ( kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) {
      if ( kdtree.radiusSearch(searchPoint, radius, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ) {
        std::vector<int> polygon_idx2use;
        for (int t = 0; t < pointIdxNKNSearch.size (); t++) {
          polygon_idx2use.push_back(polygon_raster.polygon_raster_idx[pointIdxNKNSearch[t]]);
        }
        std::set<int> s(polygon_idx2use.begin(), polygon_idx2use.end());
        polygon_idx2use.assign(s.begin(), s.end());


        for (int t=0; t < polygon_idx2use.size(); t++) {
          Point v1 = Point(cloud_hull->points[polygons[polygon_idx2use[t]].vertices[0]].x,
                           cloud_hull->points[polygons[polygon_idx2use[t]].vertices[0]].y);
          Point v2 = Point(cloud_hull->points[polygons[polygon_idx2use[t]].vertices[1]].x,
                           cloud_hull->points[polygons[polygon_idx2use[t]].vertices[1]].y);
          Point v3 = Point(cloud_hull->points[polygons[polygon_idx2use[t]].vertices[2]].x,
                           cloud_hull->points[polygons[polygon_idx2use[t]].vertices[2]].y);
          if (PointInTriangle(pt, v1, v2, v3)) {
            Eigen::Vector3f p0;
            Eigen::Vector3f p1;
            Eigen::Vector3f p2;

            float x0 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[0]].x;
            float y0 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[0]].y;
            float z0 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[0]].z;
            float x1 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[1]].x;
            float y1 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[1]].y;
            float z1 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[1]].z;
            float x2 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[2]].x;
            float y2 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[2]].y;
            float z2 = cloud_hull->points[polygons[polygon_idx2use[t]].vertices[2]].z;

            p0 << x0, y0, z0;
            p1 << x1, y1, z1;
            p2 << x2, y2, z2;

            Eigen::Vector3f normal_polygon = (p1 - p0).cross(p2 - p0);
            float nx = normal_polygon[0];
            float ny = normal_polygon[1];
            float nz = normal_polygon[2];
            float length = std::sqrt(std::pow(nx, 2) + std::pow(ny, 2) + std::pow(nz, 2));
            nx /= length;
            ny /= length;
            nz /= length;

            float polygon_z = -1000;
            if (std::abs(nz) >= 0.3) {
              polygon_z  = (nx * x0 + ny * y0 + nz * z0 - nx * i - ny * j) / nz;
            }
            if ((polygon_z > max_z) & (polygon_z > bottom_z)) {
              max_z = polygon_z;
              remain_idx = polygon_idx2use[t];
            }
          }
        }
      }

      if (remain_idx >= 0) {
        polygons2use.push_back(polygons[remain_idx]);
        (*cloud_final).insert( (*cloud_final).end(), 1, pcl::PointXYZ(i, j, max_z) );
      }
    }
  }

  // std::cout << "origin size: " << polygons.size() << std::endl;
  // std::cout << "current size: " << polygons2use.size() << std::endl;
  // plot(cloud_final);
  return cloud_final;
}

Result compute_volume(const pcl::PointCloud <pcl::PointXYZ>::Ptr &cloud_refine,
                      float bottom_z, float raster_size=0.2) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
  std::vector< pcl::Vertices > polygons;

  pcl::ConcaveHull<pcl::PointXYZ> chull;
  chull.setInputCloud(cloud_refine);
  chull.setAlpha(4.0);
  chull.reconstruct (*cloud_hull, polygons);

  // -- extract the upper polygon
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final;
  if (polygons[0].vertices.size() == 3) {
    // 1. construct polygon raster point cloud for polygons
    PolygonRaster polygon_raster =  compute_polygon_raster(cloud_hull, polygons, raster_size);

    // 2. just remain those in the lower surface (use kdtree to fast the process)
    cloud_final = compute_upper_surface_cloud (cloud_refine, cloud_hull, 
                                               polygon_raster, polygons, bottom_z, raster_size);

  } else {
    for (int i=0; i < cloud_refine->size(); i++) {
      if (cloud_refine->points[i].z > bottom_z) {
        (*cloud_final).insert( (*cloud_final).end(), 1, cloud_refine->points[i] );
      }
    }
  }

  // --compute volume under the upper polygons
  float volume = 0;
  for (int i = 0; i< cloud_final->size(); i++) {
    volume += std::pow(raster_size, 2) * (cloud_final->points[i].z - bottom_z);
  }

  return Result(cloud_final, volume);
}

PolygonRaster compute_polygon_raster(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_hull,
                                     const std::vector< pcl::Vertices > &polygons,
                                     float raster_size=0.2) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_polygon_raster (new pcl::PointCloud<pcl::PointXYZ>);

  std::vector<float> vec_rx;
  std::vector<float> vec_ry;
  std::vector<int> vec_idx;

  for (int i=0; i < polygons.size(); i++) { 
    std::vector<float> vec_vx;
    std::vector<float> vec_vy;

    for (int j=0; j < polygons[i].vertices.size(); j++) {
      vec_vx.push_back(cloud_hull->points[polygons[i].vertices[j]].x);
      vec_vy.push_back(cloud_hull->points[polygons[i].vertices[j]].y);
    }

    Point v1 = Point(vec_vx[0], vec_vy[0]);
    Point v2 = Point(vec_vx[1], vec_vy[1]);
    Point v3 = Point(vec_vx[2], vec_vy[2]);


    float min_x = *std::min_element(vec_vx.begin(), vec_vx.end());
    float max_x = *std::max_element(vec_vx.begin(), vec_vx.end());
    float min_y = *std::min_element(vec_vy.begin(), vec_vy.end());
    float max_y = *std::max_element(vec_vy.begin(), vec_vy.end());

    float start_x = min_x;
    float start_y = min_y;
    if (max_x - min_x <= raster_size) {
      start_x = (min_x + max_x) / 2;
    }
    if (max_y - min_y <= raster_size) {
      start_y = (min_y + max_y) / 2;
    }

    for (float rx = start_x; rx <= max_x; rx += raster_size) {
      for (float ry = start_y; ry <= max_y; ry += raster_size) {
        Point pt(rx, ry);
        if (PointInTriangle(pt, v1, v2, v3)) {
          vec_rx.push_back(rx);
          vec_ry.push_back(ry);
          vec_idx.push_back(i);
        }
      }
    }
  }

  cloud_polygon_raster->width = vec_idx.size();
  cloud_polygon_raster->height = 1;
  cloud_polygon_raster->points.resize(vec_idx.size());

  for (int i=0; i<cloud_polygon_raster->size(); i++) {
    cloud_polygon_raster->points[i].x = vec_rx[i];
    cloud_polygon_raster->points[i].y = vec_ry[i];
    cloud_polygon_raster->points[i].z = 0;
  }

  PolygonRaster polygon_raster(cloud_polygon_raster, vec_idx);

  return polygon_raster;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr transform2cylinder(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                       float cx, float cy) {
  // ofstream file;
  // file.open ("cylinder.csv");
  // file << "theta,rho,z\n";

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinder(new pcl::PointCloud<pcl::PointXYZ>);
  cloud_cylinder->width = cloud->width;
  cloud_cylinder->height = 1;
  cloud_cylinder->points.resize(cloud->size());

  for (int i=0; i < cloud->size(); i++) {
    float x = cloud->points[i].x - cx;
    float y = cloud->points[i].y - cy;

    float theta = std::atan2(y, x) * 180 / M_PI;
    if (theta <= 0 ) {
      theta += 360;
    }
    float rho = std::sqrt(std::pow(x, 2) + std::pow(y, 2));
    float h = cloud->points[i].z;
    // file << (std::to_string(theta) + "," + std::to_string(rho) +
    //          "," + std::to_string(z) + "\n");

    cloud_cylinder->points[i].x = theta;
    cloud_cylinder->points[i].y = rho;
    cloud_cylinder->points[i].z = h;
  }
  
  // file.close();

  return cloud_cylinder;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr remove_wall(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {

  float radius = 0.5;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2filter(new pcl::PointCloud<pcl::PointXYZ>(*cloud));

  std::vector<float> vec_x, vec_y;
  for (int i=0; i < cloud2filter->size(); i++) {
    cloud2filter->points[i].z = 0;
    vec_x.push_back(cloud2filter->points[i].x);
    vec_y.push_back(cloud2filter->points[i].y);
  }
  kdtree.setInputCloud(cloud2filter);

  float max_x = *std::max_element(vec_x.begin(), vec_x.end());
  float min_x = *std::min_element(vec_x.begin(), vec_x.end());
  float max_y = *std::max_element(vec_y.begin(), vec_y.end());
  float min_y = *std::min_element(vec_y.begin(), vec_y.end());

  std::vector<int> remain_index;
  ofstream result_file;
  result_file.open ("z_desnsity.csv");
  result_file << "x,y,diff_z\n";

  for (float i=min_x; i <= max_x; i++) {
    for (float j=min_y; j <= max_y; j++) {
      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;

      pcl::PointXYZ searchPoint;
      searchPoint.x = i;
      searchPoint.y = j;
      searchPoint.z = 0;

      if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
        std::vector<float> vec_z;
        for (int m=0; m < pointIdxRadiusSearch.size(); m++) {
          vec_z.push_back(cloud->points[pointIdxRadiusSearch[m]].z);
        }
        float max_z = *std::max_element(vec_z.begin(), vec_z.end());
        float min_z = *std::min_element(vec_z.begin(), vec_z.end());
        result_file << (std::to_string(i) + "," + std::to_string(j) +
                        "," + std::to_string(max_z-min_z) + "\n");
        
        if (max_z-min_z<2.5) {
          remain_index.insert(remain_index.end(), pointIdxRadiusSearch.begin(), pointIdxRadiusSearch.end());
        }
      }
    }
  }
  result_file.close();

  std::set<int> s(remain_index.begin(), remain_index.end());
  remain_index.assign(s.begin(), s.end());

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_wall (new pcl::PointCloud<pcl::PointXYZ>(*cloud, remain_index));

  return cloud_no_wall;
}

std::vector<int> compute_edge(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_cylinder) {
  std::vector<int> edge_idx;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_theta (new pcl::PointCloud<pcl::PointXYZ>(*cloud_cylinder));

  std::vector<float> vec_theta;
  for (int i=0; i < cloud_theta->size(); i++) {
    cloud_theta->points[i].y = 0;
    cloud_theta->points[i].z = 0;

    vec_theta.push_back(cloud_theta->points[i].x);
  }

  float min_theta = *std::min_element(vec_theta.begin(), vec_theta.end());
  float max_theta = *std::max_element(vec_theta.begin(), vec_theta.end());

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud_theta);
  float radius = 0.5;

  for (float t=min_theta; t<max_theta; t++) {
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    pcl::PointXYZ searchPoint;
    searchPoint.x = t;
    searchPoint.y = 0;
    searchPoint.z = 0;

    if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
      std::vector<float> vec_z;
      for (int i=0; i<pointIdxRadiusSearch.size(); i++) {
        vec_z.push_back(cloud_cylinder->points[pointIdxRadiusSearch[i]].z);
      }
      edge_idx.push_back(pointIdxRadiusSearch[std::distance(vec_z.begin(), std::max_element(vec_z.begin(), vec_z.end()))]);
    }
  }

  return edge_idx;
}

EdgeInfo auto_compute_edge(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  std::vector<float> vec_x;
  std::vector<float> vec_y;
  for (int i=0; i<cloud->size(); i++) {
    vec_x.push_back(cloud->points[i].x);
    vec_y.push_back(cloud->points[i].y);
  }
  float min_x = *std::min_element(vec_x.begin(), vec_x.end());
  float max_x = *std::max_element(vec_x.begin(), vec_x.end());
  float min_y = *std::min_element(vec_y.begin(), vec_y.end());
  float max_y = *std::max_element(vec_y.begin(), vec_y.end());

  std::vector<float> vec_r_std;
  std::vector<std::vector<int>> vec_edge_idx;
  std::vector<float> vec_cx;
  std::vector<float> vec_cy;
  for (float cx=min_x; cx <= max_x; cx += 2.5) {
    for (float cy=min_y; cy <= max_y; cy += 2.5) {
      vec_cx.push_back(cx);
      vec_cy.push_back(cy);
      std::vector<float> vec_r;

      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinder = transform2cylinder(cloud, cx, cy);

      std::vector<int> edge_idx;
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_theta (new pcl::PointCloud<pcl::PointXYZ>(*cloud_cylinder));

      std::vector<float> vec_theta;
      for (int i=0; i < cloud_theta->size(); i++) {
        cloud_theta->points[i].y = 0;
        cloud_theta->points[i].z = 0;

        vec_theta.push_back(cloud_theta->points[i].x);
      }

      float min_theta = *std::min_element(vec_theta.begin(), vec_theta.end());
      float max_theta = *std::max_element(vec_theta.begin(), vec_theta.end());

      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      kdtree.setInputCloud(cloud_theta);
      float radius = 0.5;

      for (float t=min_theta; t<max_theta; t++) {
        std::vector<int> pointIdxRadiusSearch;
        std::vector<float> pointRadiusSquaredDistance;

        pcl::PointXYZ searchPoint;
        searchPoint.x = t;
        searchPoint.y = 0;
        searchPoint.z = 0;

        if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
          std::vector<PointPotentialEdge> PPEs;
          for (int i=0; i<pointIdxRadiusSearch.size(); i++) {
            int ppe_idx = pointIdxRadiusSearch[i];
            float ppe_rho = cloud_cylinder->points[ppe_idx].y;
            float ppe_z = cloud_cylinder->points[ppe_idx].z;
            PointPotentialEdge ppe(ppe_idx, ppe_rho, ppe_z);
            PPEs.push_back(ppe);
          }
          std::sort(PPEs.begin(), PPEs.end(), ByRhoPPE());

          float rho_start = PPEs[0].rho;
          float z_start = PPEs[0].z;
          float rho_end = PPEs.back().rho;
          float z_end = PPEs.back().z;

          float n_rho_base = z_start - z_end;
          float n_z_base = -(rho_start - rho_end);

          float n_length = std::sqrt(std::pow(n_rho_base, 2) + std::pow(n_z_base, 2));
          n_rho_base /= n_length;
          n_z_base /= n_length;

          float max_z2base = -999;
          int potential_idx = -1;
          for (int i=0; i<PPEs.size(); i++) {
            float diff_rho_ppe = PPEs[i].rho - PPEs[0].rho;
            float diff_z_ppe = PPEs[i].z - PPEs[0].z;
            float z2base_ppe = std::abs(n_rho_base * diff_rho_ppe + n_z_base * diff_z_ppe);
            PPEs[i].z2base = z2base_ppe;

            if (PPEs[i].z2base > max_z2base) {
              max_z2base = PPEs[i].z2base;
              potential_idx = PPEs[i].idx;
            }
          }

          if ((potential_idx >= PPEs.size()) && max_z2base > 1) {
            edge_idx.push_back(potential_idx);
            float r = std::sqrt(std::pow(cloud->points[edge_idx.back()].x - cx, 2) + std::pow(cloud->points[edge_idx.back()].y - cy, 2));
            vec_r.push_back(r);
          }
        }
      }

      vec_edge_idx.push_back(edge_idx);

      float sum_r = std::accumulate(vec_r.begin(), vec_r.end(), 0.0);
      float mean_r = sum_r / vec_r.size();
      std::vector<float> vec_r_diff_square(vec_r.size());
      for (int i=0; i<vec_r.size(); i++) {
        vec_r_diff_square.push_back(std::pow(vec_r[i] - mean_r, 2));
      }
      float sq_sum_r = std::accumulate(vec_r_diff_square.begin(), vec_r_diff_square.end(), 0.0);
      vec_r_std.push_back(std::sqrt(sq_sum_r / vec_r.size()));
    }
  }
  

  int selected_c_idx = std::distance(vec_r_std.begin(), std::min_element(vec_r_std.begin(), vec_r_std.end()));
  
  float cx = vec_cx[selected_c_idx];
  float cy = vec_cy[selected_c_idx];
  std::vector<int> edge_idx = vec_edge_idx[selected_c_idx];

  EdgeInfo edge_info(cx, cy, edge_idx);

  return edge_info;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr complete_back_data(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                       const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_cylinder,
                                                       const pcl::PointCloud<pcl::Normal>::Ptr &normals,
                                                       const std::vector<int> &back_indices,
                                                       const std::vector<int> &edge_idx,
                                                       float bottom_z, float cx, float cy) {
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud);
  float radius = 0.5;

  std::vector<float> vec_added_x, vec_added_y, vec_added_z;
  for (int i=0; i < edge_idx.size(); i++) {
    pcl::PointXYZ edge_point (cloud->points[edge_idx[i]].x,
                              cloud->points[edge_idx[i]].y,
                              cloud->points[edge_idx[i]].z);
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    if ( kdtree.radiusSearch(edge_point, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
      std::vector<int> vec_idx2use;
      
      for (int j=0; j < pointIdxRadiusSearch.size(); j++) {
        float x = cloud->points[pointIdxRadiusSearch[j]].x;
        float y = cloud->points[pointIdxRadiusSearch[j]].y;
        float normal_x = normals->points[pointIdxRadiusSearch[j]].normal_x;
        float normal_y = normals->points[pointIdxRadiusSearch[j]].normal_y;
        float normal_z = normals->points[pointIdxRadiusSearch[j]].normal_z;

        if (normal_z < 0) {
          normal_x = -normal_x;
          normal_y = -normal_y;
          normal_z = -normal_z;
        }

        if (normal_x * (x - cx) + normal_y * (y - cy) > 0) {
          vec_idx2use.push_back(pointIdxRadiusSearch[j]);
        }
      }

      if (vec_idx2use.size() > 0) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinder2use (new pcl::PointCloud<pcl::PointXYZ>(*cloud_cylinder, back_indices));
        pcl::PointCloud<pcl::Normal>::Ptr cloud_cylinder_normals2use (new pcl::PointCloud<pcl::Normal>(*normals, back_indices));
        for (int i_cylinder=0; i_cylinder < cloud_cylinder2use->size(); i_cylinder++) {
          cloud_cylinder2use->points[i_cylinder].y = 0;
          cloud_cylinder2use->points[i_cylinder].z = 0;
        }
        pcl::KdTreeFLANN<pcl::PointXYZ> tree_cyliner2use;
        tree_cyliner2use.setInputCloud(cloud_cylinder2use);
        float radius_cylinder = 0.5;
        pcl::PointXYZ searchPoint_cylinder (cloud_cylinder->points[edge_idx[i]].x, 0, 0);
        std::vector<int> pointIdxRadiusSearch_cylinder;
        std::vector<float> pointRadiusSquaredDistance_cylinder;

        float z_over_xy;
        if ( tree_cyliner2use.radiusSearch (searchPoint_cylinder, radius_cylinder, pointIdxRadiusSearch_cylinder, pointRadiusSquaredDistance_cylinder) > 0 ) {
          std::vector<float> vec_z_over_xy;
          for (int ii=0; ii<pointIdxRadiusSearch_cylinder.size(); ii++) {
            float normal_x_c = cloud_cylinder_normals2use->points[pointIdxRadiusSearch_cylinder[ii]].normal_x;
            float normal_y_c = cloud_cylinder_normals2use->points[pointIdxRadiusSearch_cylinder[ii]].normal_y;
            float normal_z_c = cloud_cylinder_normals2use->points[pointIdxRadiusSearch_cylinder[ii]].normal_z;

            vec_z_over_xy.push_back(std::abs(normal_z_c / std::sqrt(std::pow(normal_x_c, 2) + std::pow(normal_y_c, 2))));
          }

          std::sort(vec_z_over_xy.begin(), vec_z_over_xy.end());
          z_over_xy = vec_z_over_xy[vec_z_over_xy.size() / 2];

          // std::cout << "z_over_xy: " << z_over_xy << std::endl;
        }


        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2use (new pcl::PointCloud<pcl::PointXYZ>(*cloud, vec_idx2use));
        pcl::search::Search<pcl::PointXYZ>::Ptr tree2use = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ>> (new pcl::search::KdTree<pcl::PointXYZ>);
        pcl::PointCloud <pcl::Normal>::Ptr normals2use = compute_normals(tree2use, cloud2use);
        
        std::vector<float> vec_normal_x;
        std::vector<float> vec_normal_y;
        std::vector<float> vec_normal_z;

        for (int j=0; j < normals2use->size(); j++) {
          float normal_x = normals2use->points[j].normal_x;
          float normal_y = normals2use->points[j].normal_y;
          float normal_z = normals2use->points[j].normal_z;

          if (normal_z < 0) {
            normal_x = -normal_x;
            normal_y = -normal_y;
          }

          vec_normal_x.push_back(normal_x);
          vec_normal_y.push_back(normal_y);
        }



        std::sort(vec_normal_x.begin(), vec_normal_x.end());
        std::sort(vec_normal_y.begin(), vec_normal_y.end());
        float normal_x = vec_normal_x[vec_normal_x.size() / 2];
        float normal_y = vec_normal_y[vec_normal_y.size() / 2];

        float lx = normal_x;
        float ly = normal_y;
        float lz = - std::sqrt(std::pow(lx, 2) + std::pow(ly, 2)) / z_over_xy;

        float length = std::sqrt(std::pow(lx, 2) + std::pow(ly, 2) + std::pow(lz, 2));
        lx /= length;
        ly /= length;
        lz /= length;

        // std::cout << "nz: " << normal_z << std::endl;
        // std::cout << "lz: " << lz << std::endl;

        if (lz < 0) {
          float added_x = cloud->points[edge_idx[i]].x;
          float added_y = cloud->points[edge_idx[i]].y;
          float added_z = cloud->points[edge_idx[i]].z;
          while (added_z + lz > bottom_z + 0.01) {
            added_x += lx;
            added_y += ly;
            added_z += lz;

            vec_added_x.push_back(added_x);
            vec_added_y.push_back(added_y);
            vec_added_z.push_back(added_z);
          }
        }
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_added (new pcl::PointCloud<pcl::PointXYZ>);
  cloud_added->width = vec_added_x.size();
  cloud_added->height = 1;
  cloud_added->points.resize (cloud_added->width * cloud_added->height);

  for (int i=0; i < cloud_added->size(); i++) {
    cloud_added->points[i].x = vec_added_x[i];
    cloud_added->points[i].y = vec_added_y[i];
    cloud_added->points[i].z = vec_added_z[i];
  }

  return cloud_added;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr refine_cloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                                 float bottom_z, bool ground=true) {

  float radius = 0.5;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2filter(new pcl::PointCloud<pcl::PointXYZ>(*cloud));

  std::vector<float> vec_x, vec_y;
  for (int i=0; i < cloud2filter->size(); i++) {
    cloud2filter->points[i].z = 0;
    vec_x.push_back(cloud2filter->points[i].x);
    vec_y.push_back(cloud2filter->points[i].y);
  }
  kdtree.setInputCloud(cloud2filter);

  float max_x = *std::max_element(vec_x.begin(), vec_x.end());
  float min_x = *std::min_element(vec_x.begin(), vec_x.end());
  float max_y = *std::max_element(vec_y.begin(), vec_y.end());
  float min_y = *std::min_element(vec_y.begin(), vec_y.end());

  std::vector<int> remain_index;

  for (float i=min_x; i <= max_x; i++) {
    for (float j=min_y; j <= max_y; j++) {
      std::vector<int> pointIdxRadiusSearch;
      std::vector<float> pointRadiusSquaredDistance;

      pcl::PointXYZ searchPoint;
      searchPoint.x = i;
      searchPoint.y = j;
      searchPoint.z = 0;

      if ( kdtree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 ) {
        std::vector<float> vec_z;
        for (int m=0; m < pointIdxRadiusSearch.size(); m++) {
          vec_z.push_back(cloud->points[pointIdxRadiusSearch[m]].z);
        }

        remain_index.push_back(pointIdxRadiusSearch[std::distance(vec_z.begin(), std::max_element(vec_z.begin(), vec_z.end()))]);
      }
    }
  }

  std::set<int> s(remain_index.begin(), remain_index.end());
  remain_index.assign(s.begin(), s.end());

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_refine (new pcl::PointCloud<pcl::PointXYZ>(*cloud, remain_index));

  if (ground) {
    int cloud_size = cloud_refine->size();
    for (int i=0; i<cloud_size; i++) {
      (*cloud_refine).insert( (*cloud_refine).end(), 1, pcl::PointXYZ(cloud_refine->points[i].x, cloud_refine->points[i].y, bottom_z) );
    }
  }

  float volume_simple = 0;
  for (int i=0; i<cloud_refine->size(); i++){
    volume_simple += cloud_refine->points[i].z - bottom_z;
  }

  // std::cout << "volume simple: " << volume_simple << std::endl;
  // std::cout << "area sum simple: " << cloud_refine->size() / 2 << std::endl;

  return cloud_refine;
}
//-----------------------------------------------------------------------------------------------------//

int main (int argc, char** argv) {
  // -- set the raster_size
  float raster_size = 0.2;

  // -- read data
  std::vector<int> filenames;
  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
  pcl::PointCloud<pcl::PointXYZ>::Ptr origin_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> (argv[filenames[0]], *origin_cloud) == -1) {
    std::cout << "Cloud reading failed." << std::endl;
    return -1;
  }


  // smooth the surface
  smooth_cloud(origin_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> ("mls.pcd", *cloud) == -1) {
    std::cout << "Cloud reading failed." << std::endl;
    return -1;
  }

  std::vector<float> z;
  for (int i=0; i < cloud->size(); i++) {
    z.push_back(cloud->points[i].z);
  }

  // -- remove walls and outliers

  float bottom_z = *std::min_element(z.begin(), z.end());

  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ>> (new pcl::search::KdTree<pcl::PointXYZ>);
  cloud = remove_wall(cloud);
  cloud = remove_outliers(cloud, bottom_z);

  // -- compute normals
  pcl::PointCloud <pcl::Normal>::Ptr normals = compute_normals(tree, cloud);

  // -- segmentation
  std::vector <pcl::PointIndices> clusters;
  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg = compute_reg(tree, cloud, normals, clusters);

  // -- generate plan descriptor
  std::vector<Plan> plans = compute_plans(cloud, clusters, normals);

  // -- detect the front pile cluster (maybe there are two front pile plan)
  bool two_front = is_two_front(plans);

  std::vector<int> front_pile_indices(clusters[plans[0].index].indices);
  if (two_front) {
    front_pile_indices.insert(front_pile_indices.end(), clusters[plans[1].index].indices.begin(), clusters[plans[1].index].indices.end());
  }

  // --detect edge
  EdgeInfo edge_info = auto_compute_edge(cloud);
  float cx = edge_info.cx;
  float cy = edge_info.cy;
  std::cout << "cx: " << cx << std::endl;
  std::cout << "cy: " << cy << std::endl;
  std::vector<int> edge_idx = edge_info.edge_idx;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_edge (new pcl::PointCloud<pcl::PointXYZ> (*cloud, edge_idx));
  // plot(cloud_edge);

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cylinder = transform2cylinder(cloud, cx, cy);

  // -- complete data set (use edge, boundary, ground as constraints)
  int start = 1;
  if (two_front) {
    start = 2;
  }
  std::vector<int> back_indices(clusters[plans[start].index].indices);
  for (int i=start+1; i < plans.size(); i++) {
    back_indices.insert(back_indices.end(), clusters[plans[i].index].indices.begin(), clusters[plans[i].index].indices.end());
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_added = complete_back_data(cloud,
                                                       cloud_cylinder,
                                                       normals,
                                                       back_indices,
                                                       edge_idx,
                                                       bottom_z, cx, cy);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_full (new pcl::PointCloud<pcl::PointXYZ>);

  for (int i=0; i < cloud->size(); i++) {
    (*cloud_full).insert( (*cloud_full).end(), 1, cloud->points[i] );
  }

  for (int i=0; i < cloud_added->size(); i++) {
    (*cloud_full).insert( (*cloud_full).end(), 1, cloud_added->points[i] );
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_refine = refine_cloud(cloud_full, bottom_z);

  // -- construct triangle for the full dataset
  pcl::ConcaveHull<pcl::PointXYZ> chull;

  chull.setInputCloud(cloud_refine);
  chull.setAlpha(4.0);
  pcl::PolygonMesh triangles;
  chull.reconstruct(triangles);
  pcl::io::saveVTKFile ("mesh_refine.4.0.vtk", triangles);

  // -- compute volume
  Result result = compute_volume(cloud_refine, bottom_z, raster_size);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_final = result.cloud_final;
  float volume = result.volume;
  std::cout << "volum: " << volume << std::endl;


  // -- display the result and write the result to disk
  plot(cloud_final);

  ofstream result_file;
  result_file.open ("result.csv");
  result_file << "volume\n";
  result_file << (std::to_string(volume) + "\n");
  result_file.close();

  return 0;
}