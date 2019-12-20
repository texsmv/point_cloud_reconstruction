// #include "utils.h"
#include <iostream>
#include <fstream>
#include <algorithm>



#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>

using namespace pcl;



double computeCloudResolution(const pcl::PointCloud<PointXYZ>::ConstPtr &cloud){
    double res = 0.0;
    int n_points = 0;
    int nres;
    std::vector<int> indices(2);
    std::vector<float> sqr_distances(2);
    pcl::search::KdTree<PointXYZ> tree;
    tree.setInputCloud(cloud);

    for (size_t i = 0; i < cloud->size(); ++i){
        if (!pcl_isfinite((*cloud)[i].x))
        {
            continue;
        }
        //Considering the second neighbor since the first is the point itself.
        nres = tree.nearestKSearch(i, 2, indices, sqr_distances);
        if (nres == 2){
            res += sqrt(sqr_distances[1]);
            ++n_points;
        }
    }
    if (n_points != 0){
        res /= n_points;
    }
    return res;
}




void compute_surface_normals(pcl::PointCloud<pcl::PointXYZ>::Ptr &points, float normal_radius, pcl::PointCloud<PointNormal>::Ptr &normals_out){
    pcl::NormalEstimation<pcl::PointXYZ, PointNormal> norm_est;
    // Use a FLANN-based KdTree to perform neighborhood searches
    // norm_est.setSearchMethod(pcl::KdTreeFLANN<pcl::PointXYZRGB>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZRGB>));
    search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>() );
    norm_est.setSearchMethod(tree);
    // Specify the size of the local neighborhood to use when
    // computing the surface normals
    // norm_est.setRadiusSearch (normal_radius);
    norm_est.setKSearch(8);
    // Set the input points
    norm_est.setInputCloud (points);
    // Set the search surface (i.e., the points that will be used
    // when search for the input points’ neighbors)
    // norm_est.setSearchSurface (points);
    // Estimate the surface normals and store the result in "normals_out"
    norm_est.compute (*normals_out);
}









int main(int argc, char *argv[]){


    float hole_size;
    float mesh_distance;
    float normal_Ksearch;
    float harris_threshold;
    float harris_radius;
    float shot_radius;
    float show_pointCloud;
    float normal_level;
    float normal_scale;

    

    std::string path1;
    std::string path2 = "";

    // load config file
    std::ifstream cFile ("../config.ini");
    if (cFile.is_open())
    {
        std::string line;
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace),
                                 line.end());
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find("=");
            auto name = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);

            if(name == "hole_size")
                hole_size = stof(value);
            if(name == "mesh_distance")
                mesh_distance = stof(value);
            if(name == "path1")
                path1 = value;
            if(name == "path2")
                path2 = value;
            if(name == "normal_Ksearch")
                normal_Ksearch = stof(value);
            if(name == "harris_threshold")
                harris_threshold = stof(value);
            if(name == "harris_radius")
                harris_radius =stof(value);
            if(name == "shot_radius")
                shot_radius = stof(value);
            if(name == "normal_level")
                normal_level = stof(value);
            if(name == "normal_scale")
                normal_scale = stof(value);
            if(name == "show_pointCloud")
                show_pointCloud = stof(value);

            // std::cout << name << " " << value << '\n';
        }
        
    }
    else {
        std::cerr << "Couldn't open config file for reading.\n";
    }
    PointCloud<PointXYZ> dcloud;
    io::loadOBJFile(path1, dcloud);
    io::savePCDFile("test.pcd", dcloud);

    PointCloud<PointXYZ>::Ptr cloud(&dcloud);
    

    PointCloud<PointNormal>::Ptr pointNormals1(new PointCloud<PointNormal>());
    compute_surface_normals(cloud, normal_Ksearch, pointNormals1);

    PointCloud<Normal>::Ptr normals(new PointCloud<Normal>());
    copyPointCloud(*pointNormals1, *normals);

    for(size_t i = 0; i<pointNormals1->points.size(); ++i){
        pointNormals1->points[i].x = cloud->points[i].x;
        pointNormals1->points[i].y = cloud->points[i].y;
        pointNormals1->points[i].z = cloud->points[i].z;

        // normals->points[i].x = cloud->points[i].normal_x;
        // normals->points[i].y = cloud->points[i].normal_y;
        // normals->points[i].z = cloud->points[i].normal_z;
    }

    PointCloud<SHOT352>::Ptr shot_features1(new PointCloud<SHOT352>());
    // compute_shot_features(harris1, cloud1, normals1, shot_features1, shot_radius, keypoints_indices1);

    float resolucion = computeCloudResolution(cloud);
    float radio = 10;
    radio = resolucion * radio;
    std::cout<<"Resolucion:  "<<resolucion<<std::endl;


    SHOTEstimation<PointXYZ, Normal, SHOT352> estimation;
    search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
    estimation.setInputCloud(cloud);
    estimation.setInputNormals(normals);
    estimation.setSearchMethod(tree);
    // estimation.setIndices(keypoints_indices);
    // estimation.setKSearch(8);
    estimation.setRadiusSearch(radio);
    estimation.compute(*shot_features1);




    io::savePCDFile("test3.pcd",  *normals);
    // io::savePCDFile("test3.pcd",  *shot_features1);
    return 0;
}

