#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/ximgproc.hpp"
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
using namespace cv::ximgproc;

int main()
{
    cout<<"Hello, Sketch2Cloths!"<<endl;

    string cur_path = "/Users/manster/Desktop/Sketch2Cloths/Cloth2Edge/";

    string model_path = cur_path + "resources/model.yml.gz";
    string image_path = cur_path + "img/";
    string res_path = cur_path + "res/";

    Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(model_path);
    
    namespace fs = boost::filesystem;
    fs::path directory(image_path);

    fs::directory_iterator end_iter;

    for(fs::directory_iterator itr(directory); itr != end_iter; ++itr)
    {
        if(fs::is_regular_file(itr->path()))
        {
            string current_file = itr->path().string();
            string current_filename = current_file.substr(current_file.size()-8, 8);
            cout<<"current_file: "<<current_filename<<endl;
            if(current_filename == "DS_Store") continue;

            Mat3b src = imread(current_file);

            Mat3f fsrc;
            src.convertTo(fsrc, CV_32F, 1.0 / 255.0);

            Mat1f edges;
            pDollar->detectEdges(fsrc, edges);


            for(int i = 0; i < edges.rows; ++i)
            {
                for(int j = 0; j < edges.cols; ++j)
                {
                    edges[i][j] = (1.0 - edges[i][j]) * 255;
                }
            }

            string output_filename = res_path + current_filename;
            imwrite(output_filename, edges);

        }
    }

    return 0;
}
