#pragma once

#include <thread>
#include <future>
#include <queue>

#include <pangolin/pangolin.h>
#include <pangolin/geometry/geometry.h>
#include <pangolin/gl/glsl.h>
#include <pangolin/gl/glvbo.h>

#include <pangolin/utils/file_utils.h>
#include <pangolin/geometry/glgeometry.h>

#include "include/run_model/TextureShader.h"
#include "include/Auxiliary.h"

#include "ORBextractor.h"
#include "System.h"

#include <Eigen/SVD>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <unordered_set>

#define NEAR_PLANE 0.1
#define FAR_PLANE 20

//draw an image of the keyPoints, the new keyPoints will appear in a different color
void drawPoints(std::vector<cv::Point3d> new_points_seen) {
    //getting general settings
    std::string settingPath = Auxiliary::GetGeneralSettingsPath();
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();

    const int point_size = data["pointSize"];

    glPointSize(point_size);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (auto point: new_points_seen) {
        glVertex3f((float) (point.x), (float) (point.y), (float) (point.z)); // color the new keyPoints in a different color
    }
    std::cout << new_points_seen.size() << std::endl;

    glEnd();
}


Eigen::Matrix4f Load_Matrix(const std::string& filename)
{
    std::ifstream file;
    std::vector<std::string> row;
    std::string line, word, temp;

    Eigen::Matrix4f mat; 

    mat(0, 0) = 6.28;
    mat(0, 1) = -0.303684;
    mat(0, 2) = -1.4316096;
    mat(0, 3) = -0.3;
    mat(1, 0) = -0.441152;
    mat(1, 1) = -6.28;
    mat(1, 2) = -0.998604;
    mat(1, 3) = 2.0;
    mat(2, 0) = -1.2091744;
    mat(2, 1) = 1.080456;
    mat(2, 2) = -6.28;
    mat(2, 3) = 2.0;
    mat(3, 0) = 0.0;
    mat(3, 1) = 0.0;
    mat(3, 2) = 0.0;
    mat(3, 3) = 1.0;

    return mat;
    /*
    file.open(filename);
    
    while (!file.eof()) {
        row.clear();

        std::getline(file, line);

        std::stringstream words(line);

        if (line == "") {
            continue;
        }

        while (std::getline(words, word, ',')) {
            try
            {
                std::stod(word);
            }
            catch(std::out_of_range)
            {
                word = "0";
            }
            row.push_back(word);
        }
        points.push_back(cv::Point3d(std::stod(row[0]), std::stod(row[1]), std::stod(row[2])));
    }
    pointData.close();
    
    return mat;
    */
}


cv::Point3d transformPoint(const cv::Point3d &point, const Eigen::Matrix4f &transformation) {
    Eigen::Vector4f eigenPoint = Eigen::Vector4f((float)point.x, (float)point.y, (float)point.z, 1.0f);
    Eigen::Vector4f transformedPoint = transformation * eigenPoint;
    return cv::Point3d((double)transformedPoint(0), (double)transformedPoint(1), (double)transformedPoint(2));
}

cv::Vec3f transformRotation(const cv::Vec3f &point, const Eigen::Matrix3f &rotation) {
    Eigen::Vector3f eigenPoint = Eigen::Vector3f((float)point[0], (float)point[1], (float)point[2]);
    Eigen::Vector3f transformedPoint = rotation * eigenPoint;
    return cv::Vec3f((double)transformedPoint(0), (double)transformedPoint(1), (double)transformedPoint(2));
}


std::vector<cv::Point3d> getVisiblePoints(const cv::Point3d& point, const cv::Vec3f& rotations, const std::string& csvfile, cv::Mat Twc)
{
    
    //getting the general settings
    std::string settingPath = Auxiliary::GetGeneralSettingsPath();
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();

    //extract the camera settings file
    std::string configPath = data["DroneYamlPathSlam"];
    cv::FileStorage fSettings(configPath, cv::FileStorage::READ);

    //camera settings
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    std::string transformation_matrix_csv_path = std::string(data["framesOutput"]) + "frames_lab_transformation_matrix.csv";

    Eigen::Matrix4f transformation_Matrix = Load_Matrix(transformation_matrix_csv_path);


    cv::Point3d transformed_Point = transformPoint(point, transformation_Matrix.inverse());

    cv::Vec3f transformed_Rotation = transformRotation(rotations, transformation_Matrix.block<3,3>(0,0).inverse());

    double yaw = rotations[0];
    double pitch = rotations[1];
    double roll = rotations[2];

    std::vector<cv::Point3d> visible_Points_slam = Auxiliary::getPointsFromPos(csvfile, transformed_Point, yaw, pitch, roll, Twc);
    
    std::vector<cv::Point3d> visible_Points_model;
    for(auto slam_point : visible_Points_slam)
    {
        visible_Points_model.push_back(transformPoint(slam_point, transformation_Matrix));
    }

    return visible_Points_model;
}

void drawVisiblePoints(const cv::Point3d& point, const cv::Vec3f& rotations, const std::string& csvfile,  cv::Mat Twc)
{
    drawPoints(getVisiblePoints(point, rotations, csvfile, Twc));
}

Eigen::Matrix4f openGlMatrixToEigen(const pangolin::OpenGlMatrix &m) {
    Eigen::Matrix4f eigen_matrix;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            eigen_matrix(row, col) = m(row, col);
        }
    }
    return eigen_matrix;
}

cv::Point3d findPos(std::shared_ptr<pangolin::OpenGlRenderState> &s_cam)
{
    const auto T_world_camera = s_cam->GetModelViewMatrix().Inverse();
    const double x = T_world_camera(0,3);
    const double y = T_world_camera(1,3);
    const double z = T_world_camera(2,3);
    return cv::Point3d((double)x, (double)y, (double)z);
}

cv::Vec3f findRotation(std::shared_ptr<pangolin::OpenGlRenderState> &s_cam)
{
    pangolin::OpenGlMatrix mv_mat = s_cam->GetModelViewMatrix();
    Eigen::Matrix4f mv_mat_eigen = openGlMatrixToEigen(mv_mat);
    Eigen::Matrix3f rotation_matrix = mv_mat_eigen.block<3, 3>(0, 0);
            
    // Convert the rotation matrix to Euler angles (yaw, pitch, roll)
    Eigen::Vector3f euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // yaw, pitch, roll
        
    // Extract transformed yaw, pitch, and roll
    float yaw = euler_angles(0), pitch = euler_angles(1), roll = euler_angles(2);
    return cv::Vec3f((double)yaw, (double)pitch, (double)roll);
}


#define NEAR_PLANE 0.1
#define FAR_PLANE 20

void applyForwardToModelCam(std::shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value);

void applyRightToModelCam(shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value);

void applyYawRotationToModelCam(std::shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value);

void applyUpModelCam(std::shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value);

void applyPitchRotationToModelCam(std::shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value);

// does the whole difference
void HandleKeyboardInput(unsigned char key, int x, int y) {
    // Handle WASD key events
    // Update camera position based on key inputs
}


// this function run the model with orb slam
void runModelAndOrbSlam(std::string &settingPath, bool *stopFlag, std::shared_ptr<pangolin::OpenGlRenderState> &s_cam,
                        bool *ready) {
    //extract the settings
    std::ifstream programData(settingPath);
    nlohmann::json data;
    programData >> data;
    programData.close();

    std::string configPath = data["DroneYamlPathSlam"];
    cv::FileStorage fSettings(configPath, cv::FileStorage::READ);

    // camera settings
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];


    std::string cloud_points = std::string(data["mapInputDir"]) + "cloud1.csv";
    
    // we initialize the camera matrix
    Eigen::Matrix3d K;
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    cv::Mat K_cv = (cv::Mat_<float>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    Eigen::Vector2i viewport_desired_size(640, 480); // resolution of image

    cv::Mat img;

    // get the parameters for the Key Points extractor
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    // Options
    bool show_bounds = false;
    bool show_axis = false;
    bool show_x0 = false;
    bool show_y0 = false;
    bool show_z0 = false;
    bool cull_backfaces = false;

    char currentDirPath[256];
    getcwd(currentDirPath, 256);
    
    //extract the information for the SLAM instance
    char time_buf[21];
    time_t now;
    std::time(&now);
    std::strftime(time_buf, 21, "%Y-%m-%d_%H:%S:%MZ", gmtime(&now));
    std::string currentTime(time_buf);
    std::string vocPath = data["VocabularyPath"];
    std::string droneYamlPathSlam = data["DroneYamlPathSlam"];
    std::string modelTextureNameToAlignTo = data["modelTextureNameToAlignTo"];
    std::string videoPath = data["offlineVideoTestPath"];
    bool loadMap = data["loadMap"];
    double movementFactor = data["movementFactor"];
    bool isSavingMap = data["saveMap"];
    std::string loadMapPath = data["loadMapPath"];
    std::string simulatorOutputDirPath = data["simulatorOutputDir"];
    std::string simulatorOutputDir = simulatorOutputDirPath + currentTime + "/";


    // Create Window for rendering
    pangolin::CreateWindowAndBind("Main", viewport_desired_size[0], viewport_desired_size[1]);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    s_cam = std::make_shared<pangolin::OpenGlRenderState>(
            pangolin::ProjectionMatrix(viewport_desired_size(0), viewport_desired_size(1), K(0, 0), K(1, 1), K(0, 2),
                                       K(1, 2), NEAR_PLANE, FAR_PLANE),
            pangolin::ModelViewLookAt(0.1, -0.1, 0.3, 0, 0, 0, 0.0, -1.0, pangolin::AxisY)); // the first 3 value are meaningless because we change them later

    // Create Interactive View in window
    pangolin::Handler3D handler(*s_cam);
    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, ((float) -viewport_desired_size[0] / (float) viewport_desired_size[1]))
            .SetHandler(&handler);

    // Load Geometry asynchronously
    std::string model_path = data["modelPath"];
    const pangolin::Geometry geom_to_load = pangolin::LoadGeometry(model_path);
    std::vector<Eigen::Vector3<unsigned int>> floorIndices;
    for (auto &o: geom_to_load.objects) {
        if (o.first == modelTextureNameToAlignTo) {
            const auto &it_vert = o.second.attributes.find("vertex_indices");
            if (it_vert != o.second.attributes.end()) {
                const auto &vs = std::get<pangolin::Image<unsigned int>>(it_vert->second);
                for (size_t i = 0; i < vs.h; ++i) {
                    const Eigen::Map<const Eigen::Vector3<unsigned int>> v(vs.RowPtr(i));
                    floorIndices.emplace_back(v);
                }
            }
        }
    }
    Eigen::MatrixXf floor(floorIndices.size() * 3, 3);
    int currentIndex = 0;
    for (const auto &b: geom_to_load.buffers) {
        const auto &it_vert = b.second.attributes.find("vertex");
        if (it_vert != b.second.attributes.end()) {
            const auto &vs = std::get<pangolin::Image<float>>(it_vert->second);
            for (auto &row: floorIndices) {
                for (auto &i: row) {
                    const Eigen::Map<const Eigen::Vector3f> v(vs.RowPtr(i));
                    floor.row(currentIndex++) = v;
                }
            }
        }
    }
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(floor, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.computeV();
    Eigen::Vector3f v = svd.matrixV().col(2);
    const auto mvm = pangolin::ModelViewLookAt(v.x(), v.y(), v.z(), 0, 0, 0, 0.0,
                                               -1.0,
                                               pangolin::AxisY);
    const auto proj = pangolin::ProjectionMatrix(viewport_desired_size(0), viewport_desired_size(1), K(0, 0), K(1, 1),
                                                 K(0, 2), K(1, 2), NEAR_PLANE, FAR_PLANE);
    s_cam->SetModelViewMatrix(mvm);
    s_cam->SetProjectionMatrix(proj);
    applyPitchRotationToModelCam(s_cam, -90);
    pangolin::GlGeometry geomToRender = pangolin::ToGlGeometry(geom_to_load);
    for (auto &buffer: geomToRender.buffers) {
        buffer.second.attributes.erase("normal");
    }
    // Render tree for holding object position
    pangolin::GlSlProgram default_prog;
    auto LoadProgram = [&]() {
        default_prog.ClearShaders();
        default_prog.AddShader(pangolin::GlSlAnnotatedShader, pangolin::shader);
        default_prog.Link();
    };
    LoadProgram();
    pangolin::RegisterKeyPressCallback('b', [&]() { show_bounds = !show_bounds; });
    pangolin::RegisterKeyPressCallback('0', [&]() { cull_backfaces = !cull_backfaces; });

    // Show axis and axis planes
    pangolin::RegisterKeyPressCallback('a', [&]() { show_axis = !show_axis; });
    pangolin::RegisterKeyPressCallback('k', [&]() { *stopFlag = !*stopFlag; });
    pangolin::RegisterKeyPressCallback('x', [&]() { show_x0 = !show_x0; });
    pangolin::RegisterKeyPressCallback('y', [&]() { show_y0 = !show_y0; });
    pangolin::RegisterKeyPressCallback('z', [&]() { show_z0 = !show_z0; });
    pangolin::RegisterKeyPressCallback('w', [&]() { applyForwardToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('a', [&]() { applyRightToModelCam(s_cam, movementFactor); });
    pangolin::RegisterKeyPressCallback('s', [&]() { applyForwardToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('d', [&]() { applyRightToModelCam(s_cam, -movementFactor); });
    pangolin::RegisterKeyPressCallback('e', [&]() { applyYawRotationToModelCam(s_cam, 1); });
    pangolin::RegisterKeyPressCallback('q', [&]() { applyYawRotationToModelCam(s_cam, -1); });
    pangolin::RegisterKeyPressCallback('r', [&]() {
        applyUpModelCam(s_cam, -movementFactor);
    });// ORBSLAM y axis is reversed
    pangolin::RegisterKeyPressCallback('f', [&]() { applyUpModelCam(s_cam, movementFactor); });
    Eigen::Vector3d Pick_w = handler.Selected_P_w();
    std::vector<Eigen::Vector3d> Picks_w;

    // opens the scan
    cv::VideoWriter writer;
    writer.open(simulatorOutputDir + "/scan.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30.0, cv::Size(viewport_desired_size[0], viewport_desired_size[1]), true);

    std::vector<cv::Point3d> seenPoints{}; // unused

    // main loop, we wont stop until we would like to 
    while (!pangolin::ShouldQuit() && !*stopFlag) {
        *ready = true;
        if ((handler.Selected_P_w() - Pick_w).norm() > 1E-6) {
            Pick_w = handler.Selected_P_w();
            Picks_w.push_back(Pick_w);
            std::cout << pangolin::FormatString("\"Translation\": [%,%,%]", Pick_w[0], Pick_w[1], Pick_w[2])
                      << std::endl;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Load any pending geometry to the GPU.
        if (d_cam.IsShown()) {
            d_cam.Activate();

            if (cull_backfaces) {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            }
            default_prog.Bind();
            default_prog.SetUniform("KT_cw", s_cam->GetProjectionMatrix() * s_cam->GetModelViewMatrix());
            pangolin::GlDraw(default_prog, geomToRender, nullptr);
            default_prog.Unbind();

            int viewport_size[4];
            glGetIntegerv(GL_VIEWPORT, viewport_size);

            pangolin::Image<unsigned char> buffer;
            pangolin::VideoPixelFormat fmt = pangolin::VideoFormatFromString("RGB24");
            buffer.Alloc(viewport_size[2], viewport_size[3], viewport_size[2] * fmt.bpp / 8);
            glReadBuffer(GL_BACK);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(0, 0, viewport_size[2], viewport_size[3], GL_RGB, GL_UNSIGNED_BYTE, buffer.ptr);

            cv::Mat imgBuffer = cv::Mat(viewport_size[3], viewport_size[2], CV_8UC3, buffer.ptr);
            cv::cvtColor(imgBuffer, img, cv::COLOR_RGB2GRAY);
            img.convertTo(img, CV_8UC1);
            cv::flip(img, img, 0);
            cv::flip(imgBuffer, imgBuffer, 0);

            writer.write(imgBuffer);

            auto now = std::chrono::system_clock::now();
            auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
            auto value = now_ms.time_since_epoch();
            double timestamp = value.count() / 1000.0;

            cv::Point3d pos = findPos(s_cam);
            cv::Vec3f rotations = findRotation(s_cam);

            drawVisiblePoints(pos, rotations, cloud_points, cv::Mat(4,4, CV_64FC1));
            s_cam->Apply();

            glDisable(GL_CULL_FACE);
        }

        pangolin::FinishFrame();
    }
    writer.release();
}



// update the camera postion when moving up
void applyUpModelCam(shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(s_cam->GetModelViewMatrix());
    camMatrix(1, 3) += value;
    s_cam->SetModelViewMatrix(camMatrix);
}

// update the camera postion when moving Forward
void applyForwardToModelCam(shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(s_cam->GetModelViewMatrix());
    camMatrix(2, 3) += value;
    s_cam->SetModelViewMatrix(camMatrix);
}

// update the camera postion when moving right
void applyRightToModelCam(shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value) {
    auto camMatrix = pangolin::ToEigen<double>(s_cam->GetModelViewMatrix());
    camMatrix(0, 3) += value;
    s_cam->SetModelViewMatrix(camMatrix);
}

// spins the camera on the y axis
void applyYawRotationToModelCam(shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value) {
    double rand = double(value) * (M_PI / 180); // M_PI isnt initialized
    double c = std::cos(rand);
    double s = std::sin(rand);

    Eigen::Matrix3d R;
    R << c, 0, s,
            0, 1, 0,
            -s, 0, c;

    Eigen::Matrix4d pangolinR = Eigen::Matrix4d::Identity();
    pangolinR.block<3, 3>(0, 0) = R;

    auto camMatrix = pangolin::ToEigen<double>(s_cam->GetModelViewMatrix());

    // Left-multiply the rotation
    camMatrix = pangolinR * camMatrix;

    // Convert back to pangolin matrix and set
    pangolin::OpenGlMatrix newModelView;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            newModelView.m[j * 4 + i] = camMatrix(i, j);
        }
    }

    s_cam->SetModelViewMatrix(newModelView);
}
// spins the camera on the x axis
void applyPitchRotationToModelCam(shared_ptr<pangolin::OpenGlRenderState> &s_cam, double value) {
    double rand = double(value) * (M_PI / 180);
    double c = std::cos(rand);
    double s = std::sin(rand);

    Eigen::Matrix3d R;
    R << 1, 0, 0,
            0, c, -s,
            0, s, c;

    Eigen::Matrix4d pangolinR = Eigen::Matrix4d::Identity();;
    pangolinR.block<3, 3>(0, 0) = R;

    auto camMatrix = pangolin::ToEigen<double>(s_cam->GetModelViewMatrix());

    // Left-multiply the rotation
    camMatrix = pangolinR * camMatrix;

    // Convert back to pangolin matrix and set
    pangolin::OpenGlMatrix newModelView;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            newModelView.m[j * 4 + i] = camMatrix(i, j);
        }
    }

    s_cam->SetModelViewMatrix(newModelView);
}


int main(int argc, char **argv) {
    //get general settings
    std::string settingPath = Auxiliary::GetGeneralSettingsPath();

    bool stopFlag = false;
    bool ready = false;
    std::shared_ptr<pangolin::OpenGlRenderState> s_cam = std::make_shared<pangolin::OpenGlRenderState>();
    std::thread t(runModelAndOrbSlam, std::ref(settingPath), &stopFlag, std::ref(s_cam), &ready); // creating a new thread to run the function

    while (!ready) {
        usleep(500);
    }

    t.join(); // waiting to the thread to finish his run
    return 0;
}
