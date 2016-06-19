cmake_minimum_required(VERSION 2.8)

INCLUDE_DIRECTORIES(${GAZETOOL_INCLUDES})

# add library install path to rpath
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


SET(GAZETOOL_SRC main.cpp)
SET(GAZETOOL_LIBS gazetool ${GAZETOOL_LIBS})

IF(ENABLE_QT_SUPPORT)
  QT5_WRAP_UI(UI_HEADERS gazergui.ui)
  SET(GAZETOOL_SRC ${GAZETOOL_SRC} gazergui.cpp glimageview.cpp ${UI_HEADERS})
  SET(GAZETOOL_LIBS ${GAZETOOL_LIBS} GL)
ENDIF()

ADD_EXECUTABLE(gazergui ${GAZETOOL_SRC})
set_target_properties(gazergui PROPERTIES OUTPUT_NAME "gazetool")
IF(ENABLE_QT_SUPPORT)
  qt5_use_modules(gazergui Core Widgets Gui OpenGL)
  ADD_DEPENDENCIES(gazergui ${UI_HEADERS})
ENDIF()
TARGET_LINK_LIBRARIES(gazergui ${GAZETOOL_LIBS})

INSTALL(TARGETS gazergui RUNTIME DESTINATION bin)

CONFIGURE_FILE("config/gazetool.sh.in" gazetool.sh @ONLY)
INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/gazetool.sh DESTINATION bin)
