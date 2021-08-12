/*
-- (c) Copyright 2018 Xilinx, Inc. All rights reserved.
--
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
--
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
--
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
--
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
*/

#include <algorithm>
#include <vector>
#include <atomic>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <mutex>
#include <zconf.h>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <dnndk/dnndk.h>

#include "utils.h"
#include "MJPEGWriter.h"

using namespace std;
using namespace cv;
using namespace std::chrono;

#define INPUT_NODE "layer0_conv"
#define KERNEL_NAME "mask_yolov3_tiny"

int idxInputImage = 0; // frame index of input video
int idxShowImage = 0;  // next frame index to be displayed
bool bReading = true;  // flag of reding input frame
chrono::system_clock::time_point start_time;

int is_video = 0;

typedef pair<int, Mat> imagePair;
class paircomp
{
public:
    bool operator()(const imagePair &n1, const imagePair &n2) const
    {
        if (n1.first == n2.first)
        {
            return (n1.first > n2.first);
        }

        return n1.first > n2.first;
    }
};

// mutex for protection of input frames queue
mutex mtxQueueInput;
// mutex for protection of display frmaes queue
mutex mtxQueueShow;
// input frames queue
queue<pair<int, Mat>> queueInput;
// display frames queue
priority_queue<imagePair, vector<imagePair>, paircomp> queueShow;

/**
 * @brief Feed input frame into DPU for process
 *
 * @param task - pointer to DPU Task for YOLO-v3 network
 * @param frame - pointer to input frame
 * @param mean - mean value for YOLO-v3 network
 *
 * @return none
 */
void setInputImageForYOLO(DPUTask *task, const Mat &frame, float *mean)
{
    Mat img_copy;
    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);
    int size = dpuGetInputTensorSize(task, INPUT_NODE);
    int8_t *data = dpuGetInputTensorAddress(task, INPUT_NODE);

    image img_new = load_image_cv(frame);
    image img_yolo = letterbox_image(img_new, width, height);

    vector<float> bb(size);
    for (int b = 0; b < height; ++b)
    {
        for (int c = 0; c < width; ++c)
        {
            for (int a = 0; a < 3; ++a)
            {
                bb[b * width * 3 + c * 3 + a] = img_yolo.data[a * height * width + b * width + c];
            }
        }
    }

    float scale = dpuGetInputTensorScale(task, INPUT_NODE);

    for (int i = 0; i < size; ++i)
    {
        data[i] = int(bb.data()[i] * scale);
        if (data[i] < 0)
            data[i] = 127;
    }

    free_image(img_new);
    free_image(img_yolo);
}

/**
 * @brief Thread entry for reading image frame from the input video file
 *
 * @param fileName - pointer to video file name
 *
 * @return none
 */
void readFrame(const char *fileName)
{
    static int loop = 3;
    VideoCapture video;
    string videoFile = fileName;
    start_time = chrono::system_clock::now();

    while (loop > 0)
    {
        loop--;
        if (!video.open(videoFile))
        {
            cout << "Fail to open specified video file:" << videoFile << endl;
            exit(-1);
        }

        while (true)
        {
            // usleep(20000);
            Mat img;
            if (queueInput.size() < 30)
            {
                if (!video.read(img))
                {
                    break;
                }

                mtxQueueInput.lock();
                queueInput.push(make_pair(idxInputImage++, img));
                mtxQueueInput.unlock();
            }
            else
            {
                usleep(10);
            }
        }

        video.release();
    }

    exit(0);
}

void readCameraFrame()
{
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
    if (!cap.isOpened())
    {
        cout << "failed to open camera" << endl;
        exit(-1);
    }
    start_time = chrono::system_clock::now();

    while (true)
    {
        // usleep(20000);
        Mat img;
        if (queueInput.size() < 30)
        {
            cap >> img;
            mtxQueueInput.lock();
            queueInput.push(make_pair(idxInputImage++, img));
            mtxQueueInput.unlock();
        }
        else
        {
            usleep(10);
        }
    }

    exit(0);
}

/**
 * @brief Thread entry for displaying image frames
 *
 * @param  none
 * @return none
 *
 */
void displayFrame()
{
    Mat frame;

    while (true)
    {
        mtxQueueShow.lock();

        if (queueShow.empty())
        {
            mtxQueueShow.unlock();
            usleep(10);
        }
        else if (idxShowImage == queueShow.top().first)
        {
            auto show_time = chrono::system_clock::now();
            stringstream buffer;
            frame = queueShow.top().second;

            auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
            buffer << fixed << setprecision(1)
                   << (float)queueShow.top().first / (dura / 1000000.f);
            string a = buffer.str() + " FPS";
            cv::putText(frame, a, cv::Point(10, 30), 1, 2, cv::Scalar{50, 240, 50}, 2);
            cv::resize(frame, frame, cv::Size(), 320.0 / frame.cols, 240.0 / frame.rows);
            cv::imshow("Yolo@Xilinx DPU", frame);

            idxShowImage++;
            queueShow.pop();
            mtxQueueShow.unlock();
            if (waitKey(1) == 'q')
            {
                bReading = false;
                exit(0);
            }
        }
        else
        {
            mtxQueueShow.unlock();
        }
    }
}

void displayFrameStream()
{
    Mat frame;
    /* Initial MJPEG server and start server ---------------------------YU */
    MJPEG server(7777);

    Mat temp_img = imread("Mask_121.jpg");
    cv::resize(temp_img, temp_img, cv::Size(), 320.0 / temp_img.cols, 240.0 / temp_img.rows);
    server.write(temp_img);
    temp_img.release();
    server.start();

    while (true)
    {
        mtxQueueShow.lock();

        if (queueShow.empty())
        {
	    //cout << "Display queue empty, DPU processing too slow!!!\n";
            mtxQueueShow.unlock();
            usleep(20);
        }
        else if (idxShowImage == queueShow.top().first)
        {
            auto show_time = chrono::system_clock::now();
            stringstream buffer;
            frame = queueShow.top().second;

            auto dura = (duration_cast<microseconds>(show_time - start_time)).count();
            buffer << fixed << setprecision(1)
                   << (float)queueShow.top().first / (dura / 1000000.f);
            string a = buffer.str() + " FPS";
	    cout << a << endl;
            cv::putText(frame, a, cv::Point(10, 30), 1, 2, cv::Scalar{50, 240, 50}, 2);
            cv::resize(frame, frame, cv::Size(), 320.0 / frame.cols, 240.0 / frame.rows);
            // cv::imshow("Yolo@Xilinx DPU", frame);
            server.write(frame);
            idxShowImage++;
            queueShow.pop();
            mtxQueueShow.unlock();
            if (waitKey(1) == 'q')
            {
                bReading = false;
                exit(0);
            }
        }
        else
        {
            mtxQueueShow.unlock();
        }
    }
}

/**
 * @brief Post process after the runing of DPU for YOLO-v3 network
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param frame
 * @param sWidth
 * @param sHeight
 *
 * @return none
 */
void postProcess(DPUTask *task, Mat &frame, int sWidth, int sHeight)
{

    /*output nodes of YOLO-v3 */
    const vector<string> outputs_node = {"layer15_conv", "layer22_conv"};
    // const vector<string> outputs_node = {"layer81_conv", "layer93_conv", "layer105_conv"};

    vector<vector<float>> boxes;
    for (size_t i = 0; i < outputs_node.size(); i++)
    {
        string output_node = outputs_node[i];
        int channel = dpuGetOutputTensorChannel(task, output_node.c_str());
        int width = dpuGetOutputTensorWidth(task, output_node.c_str());
        int height = dpuGetOutputTensorHeight(task, output_node.c_str());

        int sizeOut = dpuGetOutputTensorSize(task, output_node.c_str());
        int8_t *dpuOut = dpuGetOutputTensorAddress(task, output_node.c_str());
        float scale = dpuGetOutputTensorScale(task, output_node.c_str());
        vector<float> result(sizeOut);
        boxes.reserve(sizeOut);

        /* Store every output node results */
        get_output(dpuOut, sizeOut, scale, channel, height, width, result);

        /* Store the object detection frames as coordinate information  */
        detect(boxes, result, channel, height, width, i, sHeight, sWidth);
    }

    /* Restore the correct coordinate frame of the original image */
    correct_region_boxes(boxes, boxes.size(), frame.cols, frame.rows, sWidth, sHeight);

    /* Apply the computation for NMS */
    if (boxes.size() > 0) cout << "boxes size: " << boxes.size() << endl;
    vector<vector<float>> res = applyNMS(boxes, classificationCnt, NMS_THRESHOLD);

    float h = frame.rows;
    float w = frame.cols;
    for (size_t i = 0; i < res.size(); ++i)
    {
        float xmin = (res[i][0] - res[i][2] / 2.0) * w + 1.0;
        float ymin = (res[i][1] - res[i][3] / 2.0) * h + 1.0;
        float xmax = (res[i][0] + res[i][2] / 2.0) * w + 1.0;
        float ymax = (res[i][1] + res[i][3] / 2.0) * h + 1.0;

        //cout << res[i][res[i][4] + 6] << " ";
        //cout << xmin << " " << ymin << " " << xmax << " " << ymax << endl;

        if (res[i][res[i][4] + 6] > CONF)
        {
	    cout << res[i][res[i][4] + 6] << " ";
            cout << xmin << " " << ymin << " " << xmax << " " << ymax << endl;

            int type = res[i][4];

            if (type == 0)
            {
                //green
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 255, 0), 3, 1, 0);
            }
            else if (type == 1)
            {
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(255, 0, 0), 3, 1, 0);
            }
            else
            {
                //red
                rectangle(frame, cvPoint(xmin, ymin), cvPoint(xmax, ymax), Scalar(0, 0, 255), 3, 1, 0);
            }
        }
    }
}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 * @param img 
 *
 * @return none
 */
void runYOLO(DPUTask *task, Mat &img)
{
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    /* feed input frame into DPU Task with mean value */
    setInputImageForYOLO(task, img, mean);

    /* invoke the running of DPU for YOLO-v3 */
    dpuRunTask(task);
    postProcess(task, img, width, height);
}

/**
 * @brief Thread entry for running YOLO-v3 network on DPU for acceleration
 *
 * @param task - pointer to DPU task for running YOLO-v3
 *
 * @return none
 */
void runYOLO_video(DPUTask *task)
{
    /* mean values for YOLO-v3 */
    float mean[3] = {0.0f, 0.0f, 0.0f};

    int height = dpuGetInputTensorHeight(task, INPUT_NODE);
    int width = dpuGetInputTensorWidth(task, INPUT_NODE);

    while (true)
    {
        pair<int, Mat> pairIndexImage;

        mtxQueueInput.lock();
        if (queueInput.empty())
        {
	    cout << "Input queue empty, camera caputre & image preprocessing too slow!!!\n";
            mtxQueueInput.unlock();
            if (bReading)
            {
                continue;
            }
            else
            {
                break;
            }
        }
        else
        {
            /* get an input frame from input frames queue */
            pairIndexImage = queueInput.front();
            queueInput.pop();
            mtxQueueInput.unlock();
        }
        vector<vector<float>> res;
        /* feed input frame into DPU Task with mean value */
        setInputImageForYOLO(task, pairIndexImage.second, mean);

        /* invoke the running of DPU for YOLO-v3 */
        dpuRunTask(task);

        postProcess(task, pairIndexImage.second, width, height);
        mtxQueueShow.lock();

        /* push the image into display frame queue */
        queueShow.push(pairIndexImage);
        mtxQueueShow.unlock();
    }
}

/**
 * @brief Entry for running YOLO-v3 neural network for ADAS object detection
 *
 */
int main(const int argc, const char **argv)
{

    const int thread_count = 4;

    if (argc != 3)
    {
        cout << "Usage of this exe: ./yolo image_name[string] i"
             << endl;
        cout << "Usage of this exe: ./yolo video_name[string] v"
             << endl;
        return -1;
    }

    string model = argv[2];

    if (model == "v")
    {

        /* Attach to DPU driver and prepare for running */
        dpuOpen();

        /* Load DPU Kernels for YOLO-v3 network model */
        DPUKernel *kernel = dpuLoadKernel(KERNEL_NAME);
        vector<DPUTask *> task(thread_count - 2);

        /* Create 4 DPU Tasks for YOLO-v3 network model */
        generate(task.begin(), task.end(),
                 std::bind(dpuCreateTask, kernel, 0));

        /* Spawn 6 threads:
    - 1 thread for reading video frame
    - 4 identical threads for running YOLO-v3 network model
    - 1 thread for displaying frame in monitor
    */
        array<thread, thread_count> threadsList = {
            thread(readFrame, argv[1]),
            thread(displayFrame),
            thread(runYOLO_video, task[0]),
            thread(runYOLO_video, task[1]),
            //thread(runYOLO_video, task[2]),
            //thread(runYOLO_video, task[3]),

        };

        for (int i = 0; i < thread_count; i++)
        {
            threadsList[i].join();
        }

        /* Destroy DPU Tasks & free resources */
        for_each(task.begin(), task.end(), dpuDestroyTask);

        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;
    }
    else if (model == "c")
    {

        /* Attach to DPU driver and prepare for running */
        dpuOpen();

        /* Load DPU Kernels for YOLO-v3 network model */
        DPUKernel *kernel = dpuLoadKernel(KERNEL_NAME);
        vector<DPUTask *> task(thread_count - 2);

        /* Create 4 DPU Tasks for YOLO-v3 network model */
        generate(task.begin(), task.end(),
                 std::bind(dpuCreateTask, kernel, 0));

        /* Spawn 6 threads:
        - 1 thread for reading video frame
        - 4 identical threads for running YOLO-v3 network model
        - 1 thread for displaying frame in monitor
        */
        array<thread, thread_count> threadsList = {
            thread(readCameraFrame),
            thread(displayFrame),
            thread(runYOLO_video, task[0]),
            thread(runYOLO_video, task[1]),
            //thread(runYOLO_video, task[2]),
            //thread(runYOLO_video, task[3]),
        };

        for (int i = 0; i < thread_count; i++){
            threadsList[i].join();
        }

        /* Destroy DPU Tasks & free resources */
        for_each(task.begin(), task.end(), dpuDestroyTask);

        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;
    }
    else if (model == "i")
    {

        is_video = 0;
        dpuOpen();
        Mat img = imread(argv[1]);
        DPUKernel *kernel = dpuLoadKernel(KERNEL_NAME);
        DPUTask *task = dpuCreateTask(kernel, 0);

        runYOLO(task, img);
        imwrite("result.jpg", img);
        imshow("Xilinx DPU", img);
        waitKey(0);

        dpuDestroyTask(task);
        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;
    }
    else if (model == "cs") //camera streaming
    {        
/* Initial MJPEG server and start server ---------------------------YU */
        /*MJPEG server(7777);

        Mat temp_img = imread("Mask_121.jpg");
        cv::resize(temp_img, temp_img, cv::Size(), 320.0 / temp_img.cols, 240.0 / temp_img.rows);
        server.write(temp_img);
        temp_img.release();
        server.start(); //----------move to function displayFrameStream----------*/
/*	VideoCapture cap = cv::VideoCapture(0);
        cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
        if (!cap.isOpened())
        {
	    cout << "failed to open camera" << endl;
	    exit(-1);
        } */

/* Attach to DPU driver and prepare for running */
        dpuOpen();

        /* Load DPU Kernels for YOLO-v3 network model */
        DPUKernel *kernel = dpuLoadKernel(KERNEL_NAME);
        vector<DPUTask *> task(4);

        /* Create 4 DPU Tasks for YOLO-v3 network model */
        generate(task.begin(), task.end(),
                 std::bind(dpuCreateTask, kernel, 0));

        /* Spawn 6 threads:
        - 1 thread for reading video frame
        - 4 identical threads for running YOLO-v3 network model
        - 1 thread for displaying frame in monitor
        */
        array<thread, 6> threadsList = {
            thread(readCameraFrame),
            // thread(readCameraFrame, std::ref(cap)),
            thread(displayFrameStream),
            thread(runYOLO_video, task[0]),
            thread(runYOLO_video, task[1]),
            thread(runYOLO_video, task[2]),
            thread(runYOLO_video, task[3]),
            // thread(runYOLO_video, task[4]),
            // thread(runYOLO_video, task[5])
       };

        for (int i = 0; i < thread_count; i++){
            threadsList[i].join();
        }

        /* Destroy DPU Tasks & free resources */
        for_each(task.begin(), task.end(), dpuDestroyTask);

        /* Destroy DPU Kernels & free resources */
        dpuDestroyKernel(kernel);

        /* Dettach from DPU driver & free resources */
        dpuClose();

        return 0;

    }
    else
    {
        cout << "unknow type !" << endl;
        cout << "Usage of this exe: ./yolo image_name[string] i" << endl;
        cout << "Usage of this exe: ./yolo video_name[string] v" << endl;
        cout << "Usage of this exe: ./yolo camera c" << endl;
        cout << "Usage of this exe: ./yolo camera cs" << endl;
        return -1;
    }
}

