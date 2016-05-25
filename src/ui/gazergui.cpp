#include "gazergui.h"
#include "ui_gazergui.h"

#include <QTextStream>
#include <QSlider>
#include <QCheckBox>
#include <QCoreApplication>

GazerGui::GazerGui(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::GazerGui)
{
    ui->setupUi(this);
    QList<int> sizes;
    sizes.push_back(300);
    sizes.push_back(150);
    ui->splitter->setSizes(sizes);
    ui->splitter->setContentsMargins(0, 0, 0, 0);
}

void GazerGui::setMirror(bool val)
{
    _mirror = val;
    ui->mirrorCheckBox->setChecked(val);
}

void GazerGui::setHorizGazeTolerance(double tolerance)
{
    ui->horizToleranceSlider->setValue(tolerance*10);
}

void GazerGui::setVerticalGazeTolerance(double tolerance)
{
    ui->verticalToleranceSlider->setValue(tolerance*10);
}

GazerGui::~GazerGui()
{
    delete ui;
}

void GazerGui::displayGazehyps(GazeHypsPtr gazehyps) {
    if (_mirror) {
        cv::Mat dst;
        cv::flip(gazehyps->frame, dst, 1);
        ui->frameView->setImage(dst);
    } else {
        ui->frameView->setImage(gazehyps->frame);
    }
    if (gazehyps->size() > 0) {
        ui->eyeView->setImage(gazehyps->hyps(0).pupils.faceRegion());
        ui->normEyeView->setImage(gazehyps->hyps(0).eyePatch);
    }
    QString msg;
    QTextStream out(&msg);
    out.setRealNumberPrecision(0);
    out.setRealNumberNotation(QTextStream::FixedNotation);
    out << "fps: " << gazehyps->fps << " | latency: " << gazehyps->latency
        << " ms | frame: " << gazehyps->frameCounter;
    if (!gazehyps->id.empty()) out << " | " << QString::fromStdString(gazehyps->id);
    ui->statusbar->showMessage(msg);
}

void GazerGui::setStatusmsg(std::string msg)
{
    ui->statusbar->showMessage(QString::fromStdString(msg));
}

void GazerGui::on_horizToleranceSlider_valueChanged(int value)
{
    ui->horizToleranceLabel->setText(QString::number(value/10.0));
    emit horizGazeToleranceChanged(value/10.0);
}

void GazerGui::on_mirrorCheckBox_stateChanged(int state)
{
    _mirror = (state == Qt::Checked);
}

void GazerGui::on_verticalToleranceSlider_valueChanged(int value)
{
    ui->verticalToleranceLabel->setText(QString::number(value/10.0));
    emit verticalGazeToleranceChanged(value/10.0);
}

void GazerGui::on_smoothCheckBox_stateChanged(int state)
{
    emit smoothingChanged(state == Qt::Checked);
}

WorkerAdapter::WorkerAdapter(std::shared_ptr<WorkerThread> worker_, QObject *parent)
  : QObject(parent), worker(worker_)

{
    worker->statusSignal().connect(
          [=] (std::string message) { emit statusmsg(message); }
    );
    worker->finishedSignal().connect(
          [=] (void *) { emit finished(); }
    );
    worker->imageProcessedSignal().connect(
          [=] (GazeHypsPtr gazehyps)
        {
            emit imageProcessed(gazehyps);
            QCoreApplication::processEvents();
        }
    );
}

void WorkerAdapter::process(){
    worker->process();
}

void WorkerAdapter::stop() {
    worker->stop();
}

void WorkerAdapter::setHorizGazeTolerance(double tol) {
    worker->setHorizGazeTolerance(tol);
}

void WorkerAdapter::setVerticalGazeTolerance(double tol){
    worker->setVerticalGazeTolerance(tol);
}

void WorkerAdapter::setSmoothing(bool enabled){
    worker->setSmoothing(enabled);
}
