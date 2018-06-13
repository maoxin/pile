#include <iostream>
#include <vector>
#include <stdexcept>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>

// This function displays the help
void showHelp(char * program_name) {
    std::cout << std::endl;
    std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
    std::cout << "-h:  Show this help." << std::endl;
}

// This is the main function
pcl::PointCloud<pcl::PointXYZ>::Ptr load_pc(int argc, char** argv) {
    // Fetch point cloud filename in arguments | Works with PCD and PLY files
    std::vector<int> filenames;
    bool file_is_pcd = false;

    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".ply");
    if (filenames.size () != 1)  {
        filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");
        file_is_pcd = true;
    }

    if (filenames.size() != 1) {
        showHelp (argv[0]);
        throw std::invalid_argument( "file received is neither .pcd nor .ply" );
    }

    // Load file | Works with PCD and PLY files
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    if (file_is_pcd) {
        if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
            std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
            throw std::exception();
        }
    } else {
        if (pcl::io::loadPLYFile (argv[filenames[0]], *source_cloud) < 0)  {
            std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
            throw std::exception();
        }
    }

    return source_cloud;
}

int main(int argc, char** argv) {
    if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help") ||
        argc == 1) {
        showHelp (argv[0]);
        return 0;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud = load_pc(argc, argv);

    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(source_cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod(tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    // Use all neighbors in a sphere of radius 50cm
    ne.setRadiusSearch(0.5);

    // Compute the features
    ne.compute(*cloud_normals);

    pcl::io::savePCDFileASCII ("gridn_normals.pcd", *cloud_normals);

    return 0;
}