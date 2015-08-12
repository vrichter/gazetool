SET(INPAINT_SRC
    inpaint.cpp
)

ADD_EXECUTABLE(inpainttool ${INPAINT_SRC})
TARGET_LINK_LIBRARIES(inpainttool ${OpenCV_LIBS})

ADD_EXECUTABLE(matcher match.cpp)
TARGET_LINK_LIBRARIES(matcher ${OpenCV_LIBS})
