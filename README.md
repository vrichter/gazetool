This repository contains a calibration free gaze tracking system based on freely available libraries and data sets. The system is able to estimate horizontal and vertical gaze directions as well as eye closeness. A system description will be available in:
* `Lars Schillingmann and Yukie Nagai, "Yet Another Gaze Detector: An Embodied Calibration Free System for the iCub Robot", 15th IEEE RAS Humanoids Conference on Humanoid Robots, 2015`

Please cite the above paper when using this module for your research.

## Installation

Make sure you have the following dependencies available / installed:
* QT5: http://www.qt.io/download/
* opencv 2.x.x: http://opencv.org/downloads.html
* boost: http://www.boost.org/
* dlib: http://dlib.net/
* run `getFaceAlignmentModel.sh` in the `data` directory to download dlib's face alignment model which is required for running gazetool.

Compiling
* gazetool uses cmake, thus a standard cmake configure run is required:
`mkdir build && cd build && cmake ..`
* run `make`
* run `make install`

## Running gazetool
* Run `gazetool.sh -i camera 0` to use the first webcam attached to your system
* Run `gazetool.sh -i list 0` to list all available image providers

## Library

This project provides a cmake module file which is created in `${CMAKE_INSTALL_PREFIX}/lib/cmake/gazetool` and can be used to link against the  library.

See [main.cpp](./src/ui/main.cpp) and [workerthread.h](./src/lib/workerthread.h) for a usage example.

## Build options

The following options can be passed to cmake via the -D parameter:

###### ENABLE_QT_SUPPORT=OFF

Removes the qt dependency. The demo application will run without a gui.

###### ENABLE_RSB_SUPPORT=ON

Adds dependencies:
* rsb: https://code.cor-lab.org/projects/rsb
* rst (0.13.7+): https://code.cor-lab.org/projects/rst

Adds image providers:
* rsb <scope>: receive images via rsb on `scope`
* rsb-socket <scope>: receive images via rsb on `scope`, enforce socket
  communication

Adds publisher:
* --rsb <scope>: publishes classification results via rsb

###### ENABLE_YARP_SUPPORT=ON

Adds dependencies:
* yarp: http://www.yarp.it/

Adds image provider:
* port <port>: receive images via yarp port `port/in`

Adds publisher:
* --port <port>: publishes classification results via yarp port `port/out`

###### ENABLE_ROS_SUPPORT=ON

Adds dependencies:
* ros http://www.ros.org/ packages:
  * roscpp
  * std_msgs
  * sensor_msgs
  * image_transport
  * cv_bridge

Adds image provider:
* ros <topic>: receive images via ros topic `topic`

## Technical Notes

* Sync to vblank might negatively affect performance
  * A QT bug might further limit the maximum framerate when using multiple QT GLWidgets
* Some BLAS implementations automatically use multithreading which seems to negatively affect performance in our case.
  * If openblas is used as default blas implementation: set the environment variable `OPENBLAS_NUM_THREADS=1`
* Optimization notes
 * Include architecture specific optimzation flags such as `-march=native -O3` in `CMAKE_CXX_FLAGS`
 * Enable `USE_AVX_INSTRUCTIONS`, `USE_SSE2_INSTRUCTIONS`, or `USE_SSE4_INSTRUCTIONS` if applicable (used by dlib)
 * make sure blas and lapack libraries are installed
 * gazetool should be able to process 640x480 input at 30fps on most recent machines (including notebooks)

## References

### Methods

* F. Timm and E. Barth, “Accurate eye centre localisation by means of gradients,” in Proceedings of the International Conference on Computer Vision Theory and Applications, 2011, vol. 1, pp. 125–130.

* F. Timm and E. Barth, “Accurate, fast, and robust centre localisation for images of semiconductor components,” 2011, vol. 7877, no. 0, pp. 787705–787705–10.

* https://github.com/trishume/eyeLike

* D. E. King, “Dlib-ml: A Machine Learning Toolkit,” J. Mach. Learn. Res., vol. 10, pp. 1755–1758, Dec. 2009.

### Corpora

* F. Song, X. Tan, X. Liu, and S. Chen, “Eyes closeness detection from still images with multi-scale histograms of principal oriented gradients,” Pattern Recognit., vol. 47, no. 9, pp. 2825–2838, Sep. 2014.

* B. A. Smith, Q. Yin, S. K. Feiner, and S. K. Nayar, “Gaze locking: passive eye contact detection for human-object interaction,” in Proceedings of the 26th annual ACM symposium on User interface software and technology - UIST ’13, 2013, pp. 271–280.
