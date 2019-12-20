#include <pcl/visualization/cloud_viewer.h>
#include <iostream>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/search/search.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/keypoints/narf_keypoint.h>
#include <pcl/features/narf_descriptor.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/surface/gp3.h>
#include <pcl/features/spin_image.h>
#include <cmath>
#include <pcl/registration/icp.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/shot.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/shot.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_one_to_one.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/registration/icp.h>
#include <pcl/features/fpfh.h>
#include <pcl/geometry/eigen.h>
#include <pcl/common/geometry.h>
#include <dirent.h>



using namespace pcl;
// using namespace std;

visualization::PCLVisualizer viewer("Cloud Viewer");
visualization::PCLVisualizer ICPView("ICP Viewer");
visualization::PCLVisualizer transformed("Transformed");
visualization::PCLVisualizer both("both");

template<class T>
void load_obj(std::string path, PointCloud<T>& cloud){
    io::loadOBJFile(path, cloud);
}


template<class T>
void load_pcd(std::string path, PointCloud<T>& cloud){
    io::loadPCDFile(path, cloud);
}



void compute_shot_features(PointCloud<PointXYZ>::Ptr cloud, PointCloud<PointXYZ>::Ptr surface, PointCloud<Normal>::Ptr normals, PointCloud<SHOT352>::Ptr& features, float radio, PointIndicesConstPtr& keypoints_indices){
    SHOTEstimation<PointXYZ, Normal, SHOT352> estimation;
    search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>());
    estimation.setInputCloud(surface);
    estimation.setInputNormals(normals);
    estimation.setSearchMethod(tree);
    estimation.setIndices(keypoints_indices);
    // estimation.setKSearch(radio);
    estimation.setRadiusSearch(radio);
    // estimation.setSearchSurface(surface);
    estimation.compute(*features);
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


void detect_sift_keypoints(PointCloud<PointNormal>::Ptr &points, float min_scale, int nr_octaves, int nr_scales_per_octave, float min_contrast, PointCloud<PointWithScale>::Ptr &keypoints_out){
    SIFTKeypoint<PointNormal, PointWithScale> sift_detect;
    // Use a FLANN-based KdTree to perform neighborhood searches
    // KdTree
    

    //sift_detect.setSearchMethod(KdTreeFLANN<PointXYZRGB>::Ptr(new KdTreeFLANN<PointXYZRGB>));
    search::KdTree<PointNormal>::Ptr tree(new search::KdTree<PointNormal>() );
    // search::KdTree<PointXYZRGB>tree = new search::KdTree<PointXYZRGB>();
    sift_detect.setSearchMethod(tree);
    // sift_detect.setSearchMethod(KdTree<PointXYZRGB>::Ptr(new KdTree<PointXYZRGB>));
    // Set the detection parameters
    sift_detect.setScales (min_scale, nr_octaves, nr_scales_per_octave);
    sift_detect.setMinimumContrast (min_contrast);
    // sift_detect.setKSearch(20);
    // Set the input
    sift_detect.setInputCloud (points);
    

    

    // Detect the keypoints and store them in "keypoints_out"
    sift_detect.compute (*keypoints_out);

    std::cout << "Sift keypoints: " << keypoints_out->points.size () <<std::endl;
    
}


void detect_harris_keypoints(PointCloud<PointXYZ>::Ptr &points, PointCloud<Normal>::Ptr normals, float threshold, float radius, PointCloud<PointXYZI>::Ptr &keypoints, PointIndicesConstPtr& keypoints_indices){
    HarrisKeypoint3D <pcl::PointXYZ, pcl::PointXYZI> detector;

    detector.setNonMaxSupression (true);
    detector.setInputCloud (points);
    detector.setNormals(normals);
    detector.setRadius(0.05);
    // detector.setRadius(10);
    detector.setRefine(false);
    detector.setThreshold (threshold);
    detector.compute (*keypoints);
    keypoints_indices = detector.getKeypointsIndices();    
    pcl::console::print_highlight ("Detected %zd harris points\n", keypoints->size ());

}

void compute_PFH_features_at_keypoints(PointCloud<PointXYZRGB>::Ptr &points, PointCloud<Normal>::Ptr &normals, PointCloud<PointWithScale>::Ptr &keypoints, float feature_radius, PointCloud<PFHSignature125>::Ptr &descriptors_out){
    // Create a PFHEstimation object
    PFHEstimation<PointXYZRGB, Normal, PFHSignature125> pfh_est;
    // Set it to use a FLANN-based KdTree to perform its
    // neighborhood searches
    pfh_est.setSearchMethod(search::KdTree<PointXYZRGB>::Ptr (new search::KdTree<PointXYZRGB>));
    // Specify the radius of the PFH feature
    pfh_est.setRadiusSearch (feature_radius);
    // ... continued from the previous slide
    // Convert the keypoints cloud from PointWithScale to PointXYZRGB
    // so that it will be compatible with our original point cloud
    PointCloud<PointXYZRGB>::Ptr keypoints_xyzrgb(new PointCloud<PointXYZRGB>);
    pcl::copyPointCloud (*keypoints,*keypoints_xyzrgb);
    // Use all of the points for analyzing the local structure of the cloud
    pfh_est.setSearchSurface (points);
    pfh_est.setInputNormals (normals);// But only compute features at the keypoints
    pfh_est.setInputCloud (keypoints_xyzrgb);
    // Compute the features
    pfh_est.compute (*descriptors_out);
}


void create_hole(PointCloud<PointXYZ>::Ptr cloud, float radio){
    PointIndices::Ptr indices(new PointIndices());
    ExtractIndices<PointXYZ> extract;
    int rand_number = (rand() % (cloud->size()));
    for (size_t i = 0; i < cloud->points.size(); i++){
        if(pcl::geometry::distance(cloud->points[rand_number], cloud->points[i]) < radio){
            indices->indices.push_back(i);
            // cout<<i<<endl;
        }
    }
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setNegative(true);
    extract.filter(*cloud);

    
}


void doReconstruction(std::string path, std::string path2, float hole_size, float mesh_distance, float normal_Ksearch,    float harris_threshold,     float harris_radius,     float shot_radius,     float show_pointCloud,     float normal_level,     float normal_scale){
    


    PointCloud<PointXYZ> dcloud1;
    PointCloud<PointXYZ> dcloud2;
    load_obj<PointXYZ>(path, dcloud1);
    load_obj<PointXYZ>(path2, dcloud2);

    PointCloud<PointXYZ>::Ptr cloud1(&dcloud1);
    PointCloud<PointXYZ>::Ptr cloud2(&dcloud2);

    float resolucion = computeCloudResolution(cloud1);
    std::cout<<"Resolucion:  "<<resolucion<<std::endl;

    

    if(show_pointCloud == 1){
        visualization::PCLVisualizer show("show");
        visualization::PointCloudColorHandlerCustom<PointXYZ> cloud_handler(cloud2, 0, 0, 0);
        show.setBackgroundColor( 1.0, 1.0, 1.0 );
        show.addPointCloud<PointXYZ>(cloud2, cloud_handler, "cloud");
        show.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
        // show.initCameraParameters();
        // show.addCoordinateSystem(resolucion * 10);
        // ICPView.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Final_cloud");
        while (!show.wasStopped())
        {
            show.spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
    }



    harris_radius = resolucion * harris_radius;
    shot_radius = resolucion * shot_radius;
    hole_size = resolucion * hole_size;
    mesh_distance = resolucion * mesh_distance;

    std::cout<<"harris radius: "<<harris_radius<<std::endl;
    std::cout<<"shot radius: "<<shot_radius<<std::endl;
    std::cout<<"hole size: "<<hole_size<<std::endl;



    create_hole(cloud2, hole_size);
    create_hole(cloud2, hole_size);

    for (size_t i = 0; i < cloud2->points.size(); i++)
    {
        cloud2->points[i].x += mesh_distance;
    }

    
    // Calculate Normals
    PointCloud<PointNormal>::Ptr pointNormals1(new PointCloud<PointNormal>());
    PointCloud<PointNormal>::Ptr pointNormals2(new PointCloud<PointNormal>());
    compute_surface_normals(cloud1, normal_Ksearch, pointNormals1);
    compute_surface_normals(cloud2, normal_Ksearch, pointNormals2);
    PointCloud<Normal>::Ptr normals1(new PointCloud<Normal>());
    PointCloud<Normal>::Ptr normals2(new PointCloud<Normal>());
    copyPointCloud(*pointNormals1, *normals1);
    copyPointCloud(*pointNormals2, *normals2);
    for(size_t i = 0; i<pointNormals1->points.size(); ++i){
        pointNormals1->points[i].x = cloud1->points[i].x;
        pointNormals1->points[i].y = cloud1->points[i].y;
        pointNormals1->points[i].z = cloud1->points[i].z;
    }
    for(size_t i = 0; i<pointNormals2->points.size(); ++i){
        pointNormals2->points[i].x = cloud2->points[i].x;
        pointNormals2->points[i].y = cloud2->points[i].y;
        pointNormals2->points[i].z = cloud2->points[i].z;
    }



    // Calculate harris keypoints    
    PointCloud<PointXYZI>::Ptr harris_result1 (new PointCloud<PointXYZI>);
    PointCloud<PointXYZI>::Ptr harris_result2 (new PointCloud<PointXYZI>);
    PointCloud<PointXYZ>::Ptr harris1 (new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr harris2 (new PointCloud<PointXYZ>);

    PointIndicesConstPtr keypoints_indices1;
    PointIndicesConstPtr keypoints_indices2;

    detect_harris_keypoints(cloud1, normals1, harris_threshold, harris_radius, harris_result1, keypoints_indices1);
    detect_harris_keypoints(cloud2, normals2, harris_threshold, harris_radius, harris_result2, keypoints_indices2);

    copyPointCloud(*harris_result1, *harris1);
    copyPointCloud(*harris_result2, *harris2);

    



    // Calculate shot features from keypoints
    PointCloud<SHOT352>::Ptr shot_features1(new PointCloud<SHOT352>());
    compute_shot_features(harris1, cloud1, normals1, shot_features1, shot_radius, keypoints_indices1);

    PointCloud<SHOT352>::Ptr shot_features2(new PointCloud<SHOT352>());
    compute_shot_features(harris2, cloud2, normals2, shot_features2, shot_radius, keypoints_indices2);



    // estimate correspondences
    registration::CorrespondenceEstimation<SHOT352, SHOT352> est;
    CorrespondencesPtr correspondences(new Correspondences());
    est.setInputSource(shot_features1);
    est.setInputTarget(shot_features2);
    est.determineCorrespondences(*correspondences);

    // Duplication rejection Duplicate
    CorrespondencesPtr correspondences_result_rej_one_to_one(new Correspondences());
    registration::CorrespondenceRejectorOneToOne corr_rej_one_to_one;
    corr_rej_one_to_one.setInputCorrespondences(correspondences);
    corr_rej_one_to_one.getCorrespondences(*correspondences_result_rej_one_to_one);


    // Correspondance rejection RANSAC

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    registration::CorrespondenceRejectorSampleConsensus<PointXYZ> rejector_sac;
    CorrespondencesPtr correspondences_filtered(new Correspondences());
    rejector_sac.setInputSource(harris1);
    rejector_sac.setInputTarget(harris2);
    rejector_sac.setInlierThreshold(2.5); // distance in m, not the squared distance
    rejector_sac.setMaximumIterations(10000);
    rejector_sac.setRefineModel(false);
    rejector_sac.setInputCorrespondences(correspondences_result_rej_one_to_one);;
    rejector_sac.getCorrespondences(*correspondences_filtered);
    correspondences.swap(correspondences_filtered);
    std::cout << "Number of correspondences: "<<correspondences->size() << " Number of filtered correspondences:  " << correspondences_filtered->size() << std::endl;
    transform = rejector_sac.getBestTransformation();   // Transformation Estimation method 1



    // Transformation Estimation method 2
    //registration::TransformationEstimationSVD<PointXYZ, PointXYZ> transformation_estimation;
    //transformation_estimation.estimateRigidTransformation(*source_keypoints, *target_keypoints, *correspondences, transform);
    // std::cout << "Estimated Transform:" << std::endl << transform << std::endl;

    // / refinement transform source using transformation matrix ///////////////////////////////////////////////////////

    PointCloud<PointXYZ>::Ptr transformed_source(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr final_output(new PointCloud<PointXYZ>);
    transformPointCloud(*cloud1, *transformed_source, transform);
    // savePCDFileASCII("Transformed.pcd", (*transformed_source));
    std::cout<<"Transformation Computed"<<std::endl;

    // viewer.setBackgroundColor (0, 0, 0);
    viewer.resetCamera();
    viewer.setBackgroundColor(1, 1, 1);
    visualization::PointCloudColorHandlerCustom<PointXYZ> handler_source_cloud(transformed_source, 0, 0, 0);
    visualization::PointCloudColorHandlerCustom<PointXYZ> handler_source_keypoints(cloud1, 255, 0, 0);


    viewer.addPointCloud<PointXYZ>(cloud1, handler_source_cloud, "source_cloud");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 4, "source_cloud");
    viewer.addPointCloud<PointXYZ>(harris1, handler_source_keypoints, "source_keypoints");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "source_keypoints");
    // viewer.addPointCloudNormals<PointXYZ, PointNormal> (cloud1, pointNormals1, normal_level, normal_scale, "normals_source" );


    visualization::PointCloudColorHandlerCustom<PointXYZ> handler_target_cloud(cloud2, 0, 0, 0);
    visualization::PointCloudColorHandlerCustom<PointXYZ> handler_target_keypoints(harris2, 0, 0, 255);


    viewer.addPointCloud<PointXYZ>(cloud2, handler_target_cloud, "target_cloud");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 4, "target_cloud");
    viewer.addPointCloud<PointXYZ>(harris2, handler_target_keypoints, "target_keypoints");
    viewer.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "target_keypoints");
    // viewer.addPointCloudNormals<PointXYZ, PointNormal> (cloud2, pointNormals2, normal_level, normal_scale, "normals_target" );
    viewer.addCorrespondences<PointXYZ>(harris1, harris2, *correspondences, "correspondences");
    // viewer.addCoordinateSystem(10);

    // visualization::PointCloudColorHandlerCustom<PointXYZ> transformed_handler(cloud1, 250, 250, 250);
    // transformed.addPointCloud(cloud1, transformed_handler, "transformed");
    transformed.setBackgroundColor(1, 1, 1);
    visualization::PointCloudColorHandlerCustom<PointXYZ> transformed_handler(transformed_source, 0, 0, 0);
    transformed.addPointCloud(transformed_source, transformed_handler, "transformed");
    transformed.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 4, "transformed");

    visualization::PointCloudColorHandlerCustom<PointXYZ> target_handler(cloud2, 0, 0, 0);
    transformed.addPointCloud(cloud2, target_handler, "target");
    transformed.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "target");



    both.setBackgroundColor(1, 1, 1);
    both.addPointCloud(cloud1, transformed_handler, "cloud1");
    // visualization::PointCloudColorHandlerCustom<PointXYZ> transformed_handler(transformed_source, 250, 250, 250);
    // transformed.addPointCloud(transformed_source, transformed_handler, "transformed");
    both.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud1");

    // visualization::PointCloudColorHandlerCustom<PointXYZ> target_handler(cloud2, 250, 250, 0);
    both.addPointCloud(cloud2, target_handler, "cloud2");
    both.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud2");



    // ICP
    IterativeClosestPoint<PointXYZ, PointXYZ> icp;
    icp.setInputSource(cloud2);
    icp.setInputTarget(transformed_source);
    icp.align(*final_output);
    std::cout << "has converged:" << icp.hasConverged() << " score: " <<
    icp.getFitnessScore() << std::endl;
    std::cout << icp.getFinalTransformation() << std::endl;




    visualization::PointCloudColorHandlerCustom<PointXYZ> handler_final_cloud(final_output, 0, 0, 0);
    visualization::PointCloudColorHandlerCustom<PointXYZ> handler_final_cloud_source(transformed_source, 0, 0, 0);
    ICPView.setBackgroundColor(1, 1, 1);
    ICPView.addPointCloud<PointXYZ>(final_output, handler_final_cloud, "Final_cloud");
    ICPView.addPointCloud<PointXYZ>(transformed_source, handler_final_cloud_source, "transformed source");
    ICPView.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 5, "Final_cloud");
    ICPView.setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 4, "transformed source");
    


    // while (!viewer.wasStopped())
    // {
    //     viewer.spinOnce(100);
    //     boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    // }
//     while (!ICPView.wasStopped())
//     {
//         ICPView.spinOnce(100);
//         boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//     }

//     while (!transformed.wasStopped())
//     {
//         transformed.spinOnce(100);
//         boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//     }

//     std::cout<<"Here/n"<<std::endl;

//    while (!both.wasStopped())
//     {
//         both.spinOnce(100);
//         boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//     }

    // std::cout<<"Done/n"<<std::endl;
    viewer.spin();
    // ICPView.spin();
    // transformed.spin();
    // both.spin();
    

    std::cout<<"Done/n"<<std::endl;

}


std::vector<std::string> get_paths(){
    std::vector<std::string> paths;
    std::string path = "/home/texs/Documents/Repositorios/point_cloud_reconstruction/data/SimplifiedManifolds";

    DIR *dpdf;
    struct dirent *epdf;
    dpdf = opendir(path.c_str());
    if (dpdf != NULL){
        while (epdf = readdir(dpdf)){
            // printf("Filename: %s",epdf->d_name);
            paths.push_back(path + "/" + epdf->d_name);
            // std::cout << epdf->d_name << std::endl;
        }
    }
    closedir(dpdf);
    return paths;
}
