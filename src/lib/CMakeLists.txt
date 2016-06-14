cmake_minimum_required(VERSION 2.8)

SET(GAZETOOL_LIB_SRC
    imageprovider.cpp
    faceparts.cpp
    pupilfinder.cpp
    eyelidlearner.cpp
    mutualgazelearner.cpp
    relativeeyelidlearner.cpp
    relativegazelearner.cpp
    verticalgazelearner.cpp
    horizontalheadposelearner.cpp
    verticalheadposelearner.cpp
    facedetectionworker.cpp
    shapedetectionworker.cpp
    gazehyps.cpp
    regressionworker.cpp
    eyepatcher.cpp
    featureextractor.cpp
    abstractlearner.cpp
    rlssmoother.cpp
    blockingqueue.h
    workerthread.cpp
    workerthread.h
    subject.h
    resultpublisher.h
    resultpublisher.cpp
)

IF(ENABLE_YARP_SUPPORT)
    SET(GAZETOOL_LIB_SRC ${GAZETOOL_LIB_SRC}
        yarpsupport.cpp
    )
ENDIF()

IF(ENABLE_RSB_SUPPORT)
    SET(GAZETOOL_LIB_SRC ${GAZETOOL_LIB_SRC}
        rsbsupport.h
        rsbsupport.cpp
    )
ENDIF()

IF(ENABLE_ROS_SUPPORT)
    SET(GAZETOOL_LIB_SRC ${GAZETOOL_LIB_SRC}
        rossupport.h
        rossupport.cpp
    )
ENDIF()

INCLUDE_DIRECTORIES(${GAZETOOL_INCLUDES})

ADD_LIBRARY(gazetool SHARED ${GAZETOOL_LIB_SRC})
IF(ENABLE_QT_SUPPORT)
    qt5_use_modules(gazetool Core Widgets Gui OpenGL)
ENDIF()
TARGET_LINK_LIBRARIES(gazetool ${GAZETOOL_LIBS})


set(INSTALL_INCLUDE_DIRS "${CMAKE_INSTALL_PREFIX}/include/gazetool/")

configure_file("config/gazetoolConfig.cmake.in"
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gazetoolConfig.cmake" @ONLY)
configure_file("config/gazetoolConfigVersion.cmake.in"
  "${PROJECT_BINARY_DIR}/gazetoolConfigVersion.cmake" @ONLY)

export(TARGETS gazetool
  FILE "${PROJECT_BINARY_DIR}/gazetoolTargets.cmake")
export(PACKAGE gazetool)

INSTALL(TARGETS gazetool
     LIBRARY DESTINATION lib
)

INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
    DESTINATION "include/gazetool"
    FILES_MATCHING
    PATTERN "*.h")

install(FILES
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/gazetoolConfig.cmake"
  "${PROJECT_BINARY_DIR}/gazetoolConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/gazetoolTargets.cmake"
  DESTINATION "lib/cmake/gazetool" COMPONENT dev)
