#include "faceid.h"

#include <dlib/revision.h>

#if (DLIB_MAJOR_VERSION >= 19 && DLIB_MINOR_VERSION >= 14)
#define HAVE_DLIB_DNN
#include <dlib/dnn.h>
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#endif

#include <utility>

#ifdef HAVE_DLIB_DNN
namespace dlib_dnn {
// this is copied from the dlib-dnn_face_recognition_example
template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block,
          int N, template <typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<
    2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block =
    BN<dlib::con<N, 3, 3, 1, 1,
                 dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<
    128,
    dlib::avg_pool_everything<
        alevel0<alevel1<alevel2<alevel3<alevel4<dlib::max_pool<
            3, 3, 2, 2,
            dlib::relu<dlib::affine<dlib::con<
                32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;
} // namespace dlib_dnn

using dlib_image = dlib::matrix<dlib::rgb_pixel>;

#endif

FaceIdentificationMetric::FaceIdentificationMetric(const std::string& filename, const std::string& shape_model_filename){
#ifndef HAVE_DLIB_DNN
  std::cerr << "Faceid can be used only with dlib >= 19.14" << std::endl;
#else
  if (filename.empty()) {
    return;
  }
  auto face_recognition_model = std::make_shared<dlib_dnn::anet_type>();
  dlib::deserialize(filename) >> *face_recognition_model;
  auto shape_predictor = std::make_shared<dlib::shape_predictor>();
  dlib::deserialize(shape_model_filename) >> *shape_predictor;

  m_callback = [face_recognition_model,shape_predictor] (GazeHypList &ghyps) -> void
  {
    dlib_image face_chip;;
    for (auto ghyp : ghyps){
      dlib::extract_image_chip(ghyps.dlibimage, dlib::get_face_chip_details(ghyp.shape, 150, 0.25), face_chip);
      ghyp.faceIdVector = (*face_recognition_model)(face_chip);
    }
  };
#endif
}
