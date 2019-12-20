#include "utils.h"
#include <iostream>
#include <fstream>
#include <algorithm>






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

    std::vector<std::string> paths = get_paths();
    srand(time(0)); 
    
    if(path2 != ""){
       doReconstruction(path1, path2, hole_size, mesh_distance, normal_Ksearch, harris_threshold, harris_radius, shot_radius, show_pointCloud, normal_level, normal_scale); 
    }
    else{
        for (size_t i = 0; i < 2; i++)
        {
            int rand_number = (rand() % (paths.size()));
            // std::cout<<rand_number<<"/n";
            std::cout<<"New/n"<<std::endl;
            doReconstruction(paths[rand_number], paths[rand_number], hole_size, mesh_distance, normal_Ksearch, harris_threshold, harris_radius, shot_radius, show_pointCloud, normal_level, normal_scale);    
        }
    }

    // std::string path2 = path;

    // doReconstruction(path, path2, hole_size, mesh_distance, normal_Ksearch, harris_threshold, harris_radius, shot_radius, show_pointCloud, normal_level, normal_scale);
    
    return 0;
}

