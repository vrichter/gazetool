#pragma once
#include <QGLWidget>
#include <opencv2/opencv.hpp>

class GLImageView: public QGLWidget {
	Q_OBJECT

public:
	explicit GLImageView(QWidget* parent = 0);
	virtual ~GLImageView();
    virtual void initializeGL();
    virtual void paintGL();
    virtual void resizeGL(int width, int height);
    virtual QSize sizeHint() const;
    void setImage(const cv::Mat &frame);

private:
	cv::Mat cv_frame;
	QColor bgColor;
	GLuint texture;
    GLenum format;
    GLenum depth;

};

