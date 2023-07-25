#define CAMERA_INDEX 0


#include <memory>
#include <string>
#include <thread>
#include <iostream>
#include <unistd.h>
#include <unordered_set>
#include <nlohmann/json.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "System.h"
#include "Converter.h"
#include "include/Auxiliary.h"

/************* SIGNAL *************/
std::unique_ptr<ORB_SLAM2::System> SLAM; // an instance which helps us use the interface of ORB_SLAM
std::string simulatorOutputDir;  // stores the path of the output directory


// this function saves the descriptors, keyPoints and the Map Points
void saveMap(int mapNumber) {
    std::ofstream pointData;
    int i = 0;

    pointData.open(simulatorOutputDir + "cloud" + std::to_string(mapNumber) + ".csv");
    for (auto &p: SLAM->GetMap()->GetAllMapPoints()) {     // we go over all of our Map Points which we extract from the video
        if (p != nullptr && !p->isBad()) {
            auto point = p->GetWorldPos();  // the cordinates of our 3d Map Point as a Matrix
            Eigen::Matrix<double, 3, 1> vector = ORB_SLAM2::Converter::toVector3d(point);
            cv::Mat worldPos = cv::Mat::zeros(3, 1, CV_64F);
            worldPos.at<double>(0) = vector.x();
            worldPos.at<double>(1) = vector.y();
            worldPos.at<double>(2) = vector.z();
            p->UpdateNormalAndDepth(); // update the average devitaion of the camera from the point(normal) and update possible depth
            cv::Mat Pn = p->GetNormal(); // gets the updated normal
            Pn.convertTo(Pn, CV_64F);

            //saves the position of each Map Point and it's normal(average devitaion of the camera from the point)
            pointData << i << ",";
            pointData << worldPos.at<double>(0) << "," << worldPos.at<double>(1) << "," << worldPos.at<double>(2);
            pointData << "," << p->GetMinDistanceInvariance() << "," << p->GetMaxDistanceInvariance() << "," << Pn.at<double>(0) << "," << Pn.at<double>(1) << "," << Pn.at<double>(2);

            std::map<ORB_SLAM2::KeyFrame*, size_t> observations = p->GetObservations(); // gets all of the frames which have p
            std::ofstream keyPointsData;
            std::ofstream descriptorData;
            keyPointsData.open(simulatorOutputDir + "point" + std::to_string(i) + "_keypoints.csv");
            descriptorData.open(simulatorOutputDir + "point" + std::to_string(i) + "_descriptors.csv");
            for (auto obs : observations) { // go over all of the key Points related to this Map Point
                ORB_SLAM2::KeyFrame *currentFrame = obs.first;

                // Save keyPoints information
                cv::KeyPoint currentKeyPoint = currentFrame->mvKeys[obs.second];
                keyPointsData << currentFrame->mnId << "," << currentKeyPoint.pt.x << "," << currentKeyPoint.pt.y <<
                              "," << currentKeyPoint.size << "," << currentKeyPoint.angle << "," <<
                              currentKeyPoint.response << "," << currentKeyPoint.octave << "," <<
                              currentKeyPoint.class_id << std::endl;

                // Save Descriptors information
                cv::Mat current_descriptor = currentFrame->mDescriptors.row(obs.second);
                for (int j=0; j < current_descriptor.rows; j++) {
                    descriptorData << static_cast<int>(current_descriptor.at<uchar>(j, 0));
                    for (int k=1; k < current_descriptor.cols; k++) {
                        descriptorData << "," << static_cast<int>(current_descriptor.at<uchar>(j, k));
                    }
                    descriptorData << std::endl;
                }
            }
            keyPointsData.close();
            descriptorData.close();

            pointData << std::endl;
            i++;
        }
    }
    pointData.close();
    std::cout << "saved map" << std::endl;

}

// this function sets the behavior of the program when we get a signal s
//stops the program
void stopProgramHandler(int s) {
    saveMap(std::chrono::steady_clock::now().time_since_epoch().count());
    SLAM->Shutdown();
    cvDestroyAllWindows();
    std::cout << "stoped program" << std::endl;
    exit(1);
}

int main() 
{
    // set signal handlers
    signal(SIGINT, stopProgramHandler);
    signal(SIGTERM, stopProgramHandler);
    signal(SIGABRT, stopProgramHandler);
    signal(SIGSEGV, stopProgramHandler);

    //get genetal settings
    std::string settingPath = Auxiliary::GetGeneralSettingsPath();
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();

    //extract the information for the SLAM instance
    char time_buf[21];
    time_t now;
    std::time(&now);
    std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now));
    std::string currentTime(time_buf);
    std::string vocPath = data["VocabularyPath"];
    std::string droneYamlPathSlam = data["DroneYamlPathSlam"];
    std::string videoPath = data["offlineVideoTestPath"];
    bool loadMap = data["loadMap"];
    bool isSavingMap = data["saveMap"];
    std::string loadMapPath = data["loadMapPath"];
    std::string simulatorOutputDirPath = data["simulatorOutputDir"];
    simulatorOutputDir = simulatorOutputDirPath + currentTime + "/";
    std::filesystem::create_directory(simulatorOutputDir);
    SLAM = std::make_unique<ORB_SLAM2::System>(vocPath, droneYamlPathSlam, ORB_SLAM2::System::MONOCULAR, true, true, loadMap,
                                               loadMapPath,
                                               true); // creating the SLAM instance

    cv::Mat frame;
    int cameraIndex = CAMERA_INDEX;

    int amountOfAttepmpts = 0; // we can remove and pass 1 instead , also we can get rid of the while

    cv::VideoCapture capture(cameraIndex);
    if (!capture.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return 0;
    }

    

    // entering the main loop of the program, we will exit whem we finish to go over the video/all frames   
    for(;;)
    {
        capture.read(frame);
        if(frame.empty()){
            std::cout << "Empty" << std::endl;
            break;
        }
        SLAM->TrackMonocular(frame, capture.get(CV_CAP_PROP_POS_MSEC));

    }
    saveMap(amountOfAttepmpts); // we save our keyPoints, descriptors and Map Points
    capture.release();
    return 1;
    // we can save our map using slam
    if (isSavingMap) {
        SLAM->SaveMap(simulatorOutputDir + "simulatorMap.bin");
    }

    //we end our use of slam
    SLAM->Shutdown();
    cvDestroyAllWindows();

    return 0;
}

