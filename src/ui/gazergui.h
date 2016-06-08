#pragma once

#include <QMainWindow>
#include <QObject>

#include "../lib/gazehyps.h"
#include "../lib/workerthread.h"

namespace Ui {
    class GazerGui;
}

class GazerGui : public QMainWindow
{
    Q_OBJECT

public:
    explicit GazerGui(QWidget *parent = 0);
    void setMirror(bool val);
    void setSmoothing(bool val);
    void setHorizGazeTolerance(double tolerance);
    void setVerticalGazeTolerance(double tolerance);
    ~GazerGui();

signals:
    void horizGazeToleranceChanged(double tol);
    void verticalGazeToleranceChanged(double tol);
    bool smoothingChanged(bool enabled);

public slots:
    void displayGazehyps(GazeHypsPtr gazehyps);
    void setStatusmsg(std::string msg);

private slots:
    void on_horizToleranceSlider_valueChanged(int value);
    void on_mirrorCheckBox_stateChanged(int state);
    void on_verticalToleranceSlider_valueChanged(int value);

    void on_smoothCheckBox_stateChanged(int arg1);

private:
    Ui::GazerGui *ui;
    bool _mirror = false;
};

Q_DECLARE_METATYPE(std::string)

class WorkerAdapter : public QObject
{
    Q_OBJECT
private:
    std::shared_ptr<WorkerThread> worker;

public:
    explicit WorkerAdapter(std::shared_ptr<WorkerThread> worker, QObject *parent = 0);

signals:
    void finished();
    void imageProcessed(GazeHypsPtr gazehyps);
    void statusmsg(std::string msg);

public slots:
    void process();
    void stop();
    void setHorizGazeTolerance(double tol);
    void setVerticalGazeTolerance(double tol);
    void setSmoothing(bool enabled);
};
