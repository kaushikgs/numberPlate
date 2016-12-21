#include <string>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <cstring>
#include <boost/algorithm/string.hpp>

bool exists(std::string path);
void splitPath(std::string path, std::vector<std::string> &split);
std::vector<std::string> listDirectory(std::string dirPath, bool returnPaths);