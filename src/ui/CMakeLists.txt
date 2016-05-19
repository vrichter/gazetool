cmake_minimum_required(VERSION 2.8)

INCLUDE_DIRECTORIES(${GAZETOOL_INCLUDES})

IF(ENABLE_QT_SUPPORT)
    QT5_WRAP_UI(UI_HEADERS gazergui.ui)

    SET(GAZETOOL_BIN
        main.cpp
        gazergui.cpp
        workerthread.cpp
        glimageview.cpp
        ${UI_HEADERS})

    ADD_EXECUTABLE(gazetool_ui ${GAZETOOL_BIN})
    ADD_DEPENDENCIES(gazetool_ui ${UI_HEADERS})
    TARGET_LINK_LIBRARIES(gazetool_ui gazetool ${GAZETOOL_LIBS} GL)
    qt5_use_modules(gazetool_ui Core Widgets Gui OpenGL)

    CONFIGURE_FILE("config/gazetool.sh.in" gazetool.sh @ONLY)

    INSTALL(TARGETS gazetool_ui
      RUNTIME DESTINATION bin
    )
ENDIF()


SET(gazetool_hl
    headless.cpp
    mainloop.cpp
    )

ADD_EXECUTABLE(gazetool_hl headless.cpp mainloop.cpp)
TARGET_LINK_LIBRARIES(gazetool_hl gazetool ${GAZETOOL_LIBS})

INSTALL(TARGETS gazetool_hl
  RUNTIME DESTINATION bin
)

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/gazetool.sh DESTINATION bin)
