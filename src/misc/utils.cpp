#include "utils.h"

using namespace std;

bool exists(string path){
    struct stat info;
    stat( path.c_str(), &info);
    return S_ISDIR(info.st_mode);
}

void splitPath(string path, vector<string> &split){
    vector<string> strs;
    boost::split(strs, path, boost::is_any_of("/"));
    for(string str : strs){
        if(str != "")
            split.push_back(str);
    }
}

vector<string> listDirectory(string dirPath, bool returnPaths) {
    DIR *dir;
    struct dirent *ent;
    vector<string> result;
    
    dir = opendir(dirPath.c_str());
    if(dir == NULL){
        cout<<"Could not open Directory "<<dirPath<<endl;
        return result;
    }

    while((ent = readdir(dir)) != NULL){
        if(ent->d_type == DT_DIR)   //ignore subdirectories, datas will be there
            continue;   //corners, mser, positive, negative may be there, these shouldm't be considered as images to process
        string fileName = ent->d_name;
        if(strcmp(fileName.c_str(), ".") != 0 && strcmp(fileName.c_str(), "..") != 0 ){
            if(returnPaths){
                result.push_back(dirPath + fileName);
            }
            else{
                result.push_back(fileName);
            }
        }
    }

    return result;
}