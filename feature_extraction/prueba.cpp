#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <cctype>

using namespace std;

int main()
{
    // std::ifstream is RAII, i.e. no need to call close
    std::ifstream cFile ("config.ini");
    if (cFile.is_open())
    {
        std::string line;
        while(getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if(line[0] == '#' || line.empty())
                continue;
            size_t delimiterPos = line.find("=");
            std::string name = line.substr(0, delimiterPos);
            std::string value = line.substr(delimiterPos + 1);
            std::cout << name << " " << value << '\n';
        }
        
    }
    else {
        std::cerr << "Couldn't open config file for reading.\n";
    }
}