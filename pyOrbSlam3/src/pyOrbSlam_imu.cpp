 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>
#include <ImuTypes.h>

#include <Converter.h>
#include <MapPoint.h>

#include <fstream>
#include <iostream>
#include <string>

#include <exception>
#include <typeinfo>
#include <stdexcept>

#include "ndarray_converter.h"

#include<chrono>
#include<thread>
#include<mutex>
#include<atomic>

#include <sys/types.h>
#include <unistd.h>

namespace py = pybind11;

class Debug
{
  public:
  Debug(){};
  ~Debug(){};
  std::string getPID(){pid_t pid = getpid();
                  return to_string(pid);}
};

class PyOrbSlam
{
public:
  // Create SLAM system. It initializes all system threads and gets ready to process frames.
  ORB_SLAM3::System* slam; 
  ORB_SLAM3::Converter* conv;
  
private:
  // IMU data management
  std::vector<ORB_SLAM3::IMU::Point> imu_buffer_;
  std::mutex imu_mutex_;
  std::atomic<double> last_image_timestamp_;
  std::atomic<bool> imu_initialized_;
  ORB_SLAM3::System::eSensor sensor_type_;
  
public:
  //PyOrbSlam(){}; 
  PyOrbSlam(std::string path_to_vocabulary, std::string path_to_settings, std::string sensorType, bool useViewer)
	{
    ORB_SLAM3::System::eSensor sensor;
    if (sensorType.compare("Mono")==0){
      sensor = ORB_SLAM3::System::MONOCULAR;
    }
    if (sensorType.compare("Stereo")==0){
      sensor = ORB_SLAM3::System::STEREO;
    }
    if (sensorType.compare("RGBD")==0){
      sensor = ORB_SLAM3::System::RGBD;
    }
    if (sensorType.compare("MonoIMU")==0){
      sensor = ORB_SLAM3::System::IMU_MONOCULAR;
    }
    if (sensorType.compare("StereoIMU")==0){
      sensor = ORB_SLAM3::System::IMU_STEREO;
    }
    if (sensorType.compare("RGBDIMU")==0){
      sensor = ORB_SLAM3::System::IMU_RGBD;
    }
    
    sensor_type_ = sensor;
    last_image_timestamp_.store(-1.0);
    imu_initialized_.store(false);
    
        try{
		  slam = new ORB_SLAM3::System(path_to_vocabulary,path_to_settings,sensor, useViewer);
      conv = new ORB_SLAM3::Converter();
      
      }
      catch (const std::exception& e)
      {
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        //std::cerr << e.what() << std::endl;
        throw runtime_error(e.what());
      }
  };

  ~PyOrbSlam(){
    if (slam) {
        slam->Shutdown();
        this_thread::sleep_for(chrono::milliseconds(5000));
        delete slam;
        slam = nullptr;
    }
    if (conv) {
        delete conv;
        conv = nullptr;
    }
  };

  void Shutdown(){
     try{
        slam->Shutdown();
     }
    catch (const std::exception& e)
      {
        cout << e.what() << endl;
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        //std::cerr << e.what() << std::endl;
        throw runtime_error(e.what());
      }
  };

  void saveTrajectory(string filePath){
    // Save camera trajectory
    slam->SaveKeyFrameTrajectoryTUM(filePath);
  };

  void ActivateLocalizationMode(){slam->ActivateLocalizationMode();};
    
  void DeactivateLocalizationMode(){slam->DeactivateLocalizationMode();};

  void Reset(){slam->Reset();};

  void ResetActiveMap(){slam->ResetActiveMap();};

  int GetTrackingState(){return slam->GetTrackingState();};

  bool IsLost(){return slam->isLost();};

  long unsigned int getCurrentMapID(){
      ORB_SLAM3::Map* pActiveMap;
      pActiveMap = slam->mpAtlas->GetCurrentMap();
      return pActiveMap->GetId();
  };

  int getNrOfMaps(){
      vector<ORB_SLAM3::Map*> vpMaps = slam->mpAtlas->GetAllMaps();
      return vpMaps.size();
  }

  cv::Mat GetTrackedMapReferencePointsOfMap(int mapNr){
    try{
      ORB_SLAM3::Map* pActiveMap;
      if (mapNr == -1){
        pActiveMap = slam->mpAtlas->GetCurrentMap();
      }
      else {
        vector<ORB_SLAM3::Map*> vpMaps = slam->mpAtlas->GetAllMaps();
        pActiveMap = vpMaps[mapNr];
      }
      if(!pActiveMap)
        return  cv::Mat(1,3,CV_32FC1, 0.0f);
    const vector<ORB_SLAM3::MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();
    set<ORB_SLAM3::MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());
    if(vpRefMPs.empty())
        return cv::Mat(1,3,CV_32FC1, 0.0f);
    cv::Mat positions = cv::Mat(vpRefMPs.size(),40,CV_32FC1, 0.0f);
    for(size_t i=0, iend=vpRefMPs.size(); i<iend;i++)
      {
          if(vpRefMPs[i]->isBad())
              continue;
          Eigen::Matrix<float,3,1> pos = vpRefMPs[i]->GetWorldPos();
          //glVertex3f(pos(0),pos(1),pos(2));
          positions.at<float>(i,0) = pos(0);
          positions.at<float>(i,1) = pos(1);
          positions.at<float>(i,2) = pos(2);
          Eigen::Matrix<float,3,1> norm = vpRefMPs[i]->GetNormal();
          //glVertex3f(pos(0),pos(1),pos(2));
          positions.at<float>(i,3) = norm(0);
          positions.at<float>(i,4) = norm(1);
          positions.at<float>(i,5) = norm(2);
          positions.at<float>(i,6) = float(vpRefMPs[i]->Observations());
          positions.at<float>(i,7) = vpRefMPs[i]->GetFoundRatio();
          cv::Mat descr =  vpRefMPs[i]->GetDescriptor();
          for (int z = 0; z<32; z++){
            positions.at<float>(i,8+z) = float(descr.at<unsigned char>(z));
          }
      }
    return positions;
    }
    catch (const std::exception& e)
      {
        cout << e.what() << endl;
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        //std::cerr << e.what() << std::endl;
        throw runtime_error(e.what());
      }
  }


    cv::Mat GetTrackedMapPointsOfMap(int mapNr){
    try{
      ORB_SLAM3::Map* pActiveMap;
      if (mapNr == -1){
        pActiveMap = slam->mpAtlas->GetCurrentMap();
      }
      else {
        vector<ORB_SLAM3::Map*> vpMaps = slam->mpAtlas->GetAllMaps();
        pActiveMap = vpMaps[mapNr];
      }
      if(!pActiveMap)
        return  cv::Mat(1,3,CV_32FC1, 0.0f);

    const vector<ORB_SLAM3::MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<ORB_SLAM3::MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    set<ORB_SLAM3::MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return cv::Mat(1,3,CV_32FC1, 0.0f);
    cv::Mat positions = cv::Mat(vpMPs.size(),40,CV_32FC1, 0.0f);
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
      {
          if(vpMPs[i]->isBad() )//|| spRefMPs.count(vpMPs[i]))
              continue;
          Eigen::Matrix<float,3,1> pos = vpMPs[i]->GetWorldPos();
          //glVertex3f(pos(0),pos(1),pos(2));
          positions.at<float>(i,0) = pos(0);
          positions.at<float>(i,1) = pos(1);
          positions.at<float>(i,2) = pos(2);
          Eigen::Matrix<float,3,1> norm = vpMPs[i]->GetNormal();
          //glVertex3f(pos(0),pos(1),pos(2));
          positions.at<float>(i,3) = norm(0);
          positions.at<float>(i,4) = norm(1);
          positions.at<float>(i,5) = norm(2);
          positions.at<float>(i,6) = float(vpMPs[i]->Observations());
          positions.at<float>(i,7) = vpMPs[i]->GetFoundRatio();
          cv::Mat descr =  vpMPs[i]->GetDescriptor();
          for (int z = 0; z<32; z++){
            positions.at<float>(i,8+z) = float(descr.at<unsigned char>(z));
          }
      }
    return positions;
    }
    catch (const std::exception& e)
      {
        cout << e.what() << endl;
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        //std::cerr << e.what() << std::endl;
        throw runtime_error(e.what());
      }
  }

  cv::Mat getKeyFramesOfMap(int mapNr, bool withIMU){
      try{
            ORB_SLAM3::Map* pActiveMap;
            if (mapNr == -1){
              pActiveMap = slam->mpAtlas->GetCurrentMap();
            }
            else {
              vector<ORB_SLAM3::Map*> vpMaps = slam->mpAtlas->GetAllMaps();
              pActiveMap = vpMaps[mapNr];
            }
            if(!pActiveMap)
              return  cv::Mat(1,3,CV_32FC1, 0.0f);
            vector<ORB_SLAM3::KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();
            sort(vpKFs.begin(),vpKFs.end(),ORB_SLAM3::KeyFrame::lId);
            cv::Mat keyPositions = cv::Mat(vpKFs.size(),7,CV_32FC1, 0.0f);
            for(size_t i=0; i<vpKFs.size(); i++)
            {
                ORB_SLAM3::KeyFrame* pKF = vpKFs[i];

              // pKF->SetPose(pKF->GetPose()*Two);

                if(!pKF || pKF->isBad())
                    continue;
                if (withIMU)
                {
                    Sophus::SE3f Twb = pKF->GetImuPose();
                    Eigen::Quaternionf q = Twb.unit_quaternion();
                    Eigen::Vector3f twb = Twb.translation();
                    keyPositions.at<float>(i,0) = twb(0);
                    keyPositions.at<float>(i,1) = twb(1);
                    keyPositions.at<float>(i,2) = twb(2);
                    keyPositions.at<float>(i,3) = q.x();
                    keyPositions.at<float>(i,4) = q.y();
                    keyPositions.at<float>(i,5) = q.z();
                    keyPositions.at<float>(i,6) = q.w();
                }
                else
                {
                    Sophus::SE3f Twc = pKF->GetPoseInverse();
                    Eigen::Quaternionf q = Twc.unit_quaternion();
                    Eigen::Vector3f t = Twc.translation();
                    keyPositions.at<float>(i,0) = t(0);
                    keyPositions.at<float>(i,1) = t(1);
                    keyPositions.at<float>(i,2) = t(2);
                    keyPositions.at<float>(i,3) = q.x();
                    keyPositions.at<float>(i,4) = q.y();
                    keyPositions.at<float>(i,5) = q.z();
                    keyPositions.at<float>(i,6) = q.w();
                }
            }
            return keyPositions;
          }
          catch (const std::exception& e)
      {
        cout << e.what() << endl;
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        //std::cerr << e.what() << std::endl;
        throw runtime_error(e.what());
      }
  }
  
  cv::Mat getFramePoints(){
    try{
      ORB_SLAM3::LocalSaveClass& lsave = ORB_SLAM3::LocalSaveClass::getInstance();
      return lsave.getFramePoints();
    }
    catch (const std::exception& e)
      {
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        throw  runtime_error(e.what());
        //std::cerr << e.what() << std::endl;
      }
  }
  
  // ==============================================================
  // IMU-ENHANCED METHODS
  // ==============================================================
  
  // Convert NumPy array to vector of IMU measurements
  std::vector<ORB_SLAM3::IMU::Point> convertIMUData(py::array_t<float> input) {
    std::vector<ORB_SLAM3::IMU::Point> imu_measurements;
    auto buf = input.request();
    
    if (buf.ndim != 2 || buf.shape[1] != 7) {
        throw std::runtime_error("IMU data must be (N, 7) array: [timestamp, ax, ay, az, wx, wy, wz]");
    }
    
    float *ptr = (float *) buf.ptr;
    
    for (int i = 0; i < buf.shape[0]; i++) {
        double timestamp = ptr[i*7 + 0];
        float ax = ptr[i*7 + 1], ay = ptr[i*7 + 2], az = ptr[i*7 + 3];
        float wx = ptr[i*7 + 4], wy = ptr[i*7 + 5], wz = ptr[i*7 + 6];
        
        imu_measurements.emplace_back(ax, ay, az, wx, wy, wz, timestamp);
    }
    return imu_measurements;
  }
  
  // Filter IMU measurements between two timestamps
  std::vector<ORB_SLAM3::IMU::Point> filterIMUMeasurements(double start_time, double end_time) {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    std::vector<ORB_SLAM3::IMU::Point> filtered;
    
    for (const auto& imu_point : imu_buffer_) {
        if (imu_point.t > start_time && imu_point.t <= end_time) {
            filtered.push_back(imu_point);
        }
    }
    return filtered;
  }
  
  // Add single IMU measurement to buffer
  void addIMUMeasurement(float ax, float ay, float az, float wx, float wy, float wz, double timestamp) {
    try {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buffer_.emplace_back(ax, ay, az, wx, wy, wz, timestamp);
        
        // Keep buffer size reasonable (remove old measurements)
        const size_t MAX_BUFFER_SIZE = 10000;
        if (imu_buffer_.size() > MAX_BUFFER_SIZE) {
            imu_buffer_.erase(imu_buffer_.begin(), imu_buffer_.begin() + 1000);
        }
    } catch (const std::exception& e) {
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        throw runtime_error(e.what());
    }
  }
  
  // Add batch of IMU measurements to buffer
  void addIMUMeasurements(py::array_t<float> imu_data) {
    try {
        auto measurements = convertIMUData(imu_data);
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buffer_.insert(imu_buffer_.end(), measurements.begin(), measurements.end());
        
        // Keep buffer size reasonable
        const size_t MAX_BUFFER_SIZE = 10000;
        if (imu_buffer_.size() > MAX_BUFFER_SIZE) {
            size_t excess = imu_buffer_.size() - MAX_BUFFER_SIZE + 1000;
            imu_buffer_.erase(imu_buffer_.begin(), imu_buffer_.begin() + excess);
        }
    } catch (const std::exception& e) {
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        throw runtime_error(e.what());
    }
  }
  
  // Clear IMU buffer
  void clearIMUBuffer() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_buffer_.clear();
    last_image_timestamp_.store(-1.0);
    imu_initialized_.store(false);
  }
  
  // Get IMU buffer size
  int getIMUBufferSize() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    return imu_buffer_.size();
  }
  
  // Check if IMU is initialized
  bool isIMUInitialized() {
    return imu_initialized_.load();
  }
  
  // Get time since IMU initialization  
  double getIMUInitTime() {
    return last_image_timestamp_.load();
  }
            
  // Original process method (maintained for backward compatibility)
  cv::Mat process(cv::Mat &in_image, const double &seconds){
    cv::Mat camPose;
    Sophus::SE3f camPoseReturn;
    g2o::SE3Quat g2oQuat;
    try{
      camPoseReturn = slam->TrackMonocular(in_image,seconds);
      g2oQuat = conv->toSE3Quat(camPoseReturn);
      camPose = conv->toCvMat(g2oQuat);
      
      // Update timestamp for IMU synchronization
      last_image_timestamp_.store(seconds);
    }
    catch (const std::exception& e)
      {
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        throw  runtime_error(e.what());
        //std::cerr << e.what() << std::endl;
      }
    return camPose;
  }
  
  // Enhanced process method with IMU data
  cv::Mat processWithIMU(cv::Mat &in_image, const double &timestamp, py::array_t<float> imu_data) {
    cv::Mat camPose;
    Sophus::SE3f camPoseReturn;
    g2o::SE3Quat g2oQuat;
    
    try {
        // Convert IMU data
        auto imu_measurements = convertIMUData(imu_data);
        
        // Call appropriate tracking method based on sensor type
        if (sensor_type_ == ORB_SLAM3::System::IMU_MONOCULAR) {
            camPoseReturn = slam->TrackMonocular(in_image, timestamp, imu_measurements);
        } else if (sensor_type_ == ORB_SLAM3::System::MONOCULAR) {
            // Fallback to monocular-only if not IMU sensor
            camPoseReturn = slam->TrackMonocular(in_image, timestamp);
        } else {
            throw std::runtime_error("Unsupported sensor type for processWithIMU");
        }
        
        g2oQuat = conv->toSE3Quat(camPoseReturn);
        camPose = conv->toCvMat(g2oQuat);
        
        // Update timestamps
        last_image_timestamp_.store(timestamp);
        if (!imu_initialized_.load() && !imu_measurements.empty()) {
            imu_initialized_.store(true);
        }
        
    } catch (const std::exception& e) {
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        throw runtime_error(e.what());
    }
    return camPose;
  }
  
  // Process with buffered IMU data (automatic filtering)
  cv::Mat processWithBufferedIMU(cv::Mat &in_image, const double &timestamp) {
    cv::Mat camPose;
    Sophus::SE3f camPoseReturn;
    g2o::SE3Quat g2oQuat;
    
    try {
        double last_timestamp = last_image_timestamp_.load();
        std::vector<ORB_SLAM3::IMU::Point> imu_measurements;
        
        if (last_timestamp > 0) {
            // Filter IMU measurements between last and current timestamp
            imu_measurements = filterIMUMeasurements(last_timestamp, timestamp);
        }
        
        // Call appropriate tracking method
        if (sensor_type_ == ORB_SLAM3::System::IMU_MONOCULAR) {
            camPoseReturn = slam->TrackMonocular(in_image, timestamp, imu_measurements);
        } else {
            camPoseReturn = slam->TrackMonocular(in_image, timestamp);
        }
        
        g2oQuat = conv->toSE3Quat(camPoseReturn);
        camPose = conv->toCvMat(g2oQuat);
        
        // Update timestamp
        last_image_timestamp_.store(timestamp);
        if (!imu_initialized_.load() && !imu_measurements.empty()) {
            imu_initialized_.store(true);
        }
        
    } catch (const std::exception& e) {
        ofstream myfile;
        myfile.open ("errorLog.txt");
        myfile << e.what();
        myfile.close();
        throw runtime_error(e.what());
    }
    return camPose;
  }
};

PYBIND11_MODULE(pyOrbSlam, m)
{
	NDArrayConverter::init_numpy();
  py::class_<Debug>(m, "Debug")
    .def(py::init())
    .def("getPID",&Debug::getPID);
    
	py::class_<PyOrbSlam>(m, "OrbSlam")
    //.def(py::init())
		.def(py::init<string,string, string,bool>(), py::arg("path_to_vocabulary"), py::arg("path_to_settings"), py::arg("sensorType")="Mono", py::arg("useViewer")=false)
		.def("saveTrajectory", &PyOrbSlam::saveTrajectory, py::arg("filePath"))
		// Original methods
		.def("process", &PyOrbSlam::process, py::arg("in_image"), py::arg("seconds"))
    .def("ActivateLocalizationMode", &PyOrbSlam::ActivateLocalizationMode)
    .def("DeactivateLocalizationMode", &PyOrbSlam::DeactivateLocalizationMode)
    .def("Reset", &PyOrbSlam::Reset)
    .def("ResetActiveMap", &PyOrbSlam::ResetActiveMap)
    .def("GetTrackingState", &PyOrbSlam::GetTrackingState)
    .def("GetCurrentMapID", &PyOrbSlam::getCurrentMapID)
    .def("IsLost", &PyOrbSlam::IsLost)
    .def("getFramePoints",&PyOrbSlam::getFramePoints)
    .def("GetTrackedMapPoints", &PyOrbSlam::GetTrackedMapPointsOfMap, py::arg("mapNr")=-1)
    .def("GetTrackedMapReferencePoints", &PyOrbSlam::GetTrackedMapReferencePointsOfMap, py::arg("mapNr")=-1)
    .def("getNrOfMaps", &PyOrbSlam::getNrOfMaps)
    .def("getKeyFramesOfMap", &PyOrbSlam::getKeyFramesOfMap, py::arg("mapNr")=-1, py::arg("withIMU") = false)
    .def("shutdown",&PyOrbSlam::Shutdown)
    // New IMU methods
    .def("processWithIMU", &PyOrbSlam::processWithIMU, 
         py::arg("in_image"), py::arg("timestamp"), py::arg("imu_data"),
         "Process frame with IMU measurements. imu_data format: (N, 7) [timestamp, ax, ay, az, wx, wy, wz]")
    .def("processWithBufferedIMU", &PyOrbSlam::processWithBufferedIMU,
         py::arg("in_image"), py::arg("timestamp"),
         "Process frame using buffered IMU data (automatic filtering)")
    .def("addIMUMeasurement", &PyOrbSlam::addIMUMeasurement,
         py::arg("ax"), py::arg("ay"), py::arg("az"), 
         py::arg("wx"), py::arg("wy"), py::arg("wz"), py::arg("timestamp"),
         "Add single IMU measurement to buffer")
    .def("addIMUMeasurements", &PyOrbSlam::addIMUMeasurements,
         py::arg("imu_data"),
         "Add batch of IMU measurements to buffer")
    .def("clearIMUBuffer", &PyOrbSlam::clearIMUBuffer,
         "Clear IMU measurement buffer")
    .def("getIMUBufferSize", &PyOrbSlam::getIMUBufferSize,
         "Get current IMU buffer size")
    .def("isIMUInitialized", &PyOrbSlam::isIMUInitialized,
         "Check if IMU tracking is initialized")
    .def("getIMUInitTime", &PyOrbSlam::getIMUInitTime,
         "Get timestamp of IMU initialization");
};