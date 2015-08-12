cd build
dlibgazer/dlibgazer -m ../data/shape_predictor_68_face_landmarks.dat -c 0 --classify ../data/mutual_gaze_classifier.dat

Notes:
Sync to vblank should be disabled, otherwise framerate will be limited due to multiple QT GLWidgets.

Warning openblas affects performance due to threading! 
OPENBLAS_NUM_THREADS=1

Sources:

[1] F. Timm and E. Barth, “Accurate eye centre localisation by means of gradients,” in Proceedings of the International Conference on Computer Vision Theory and Applications, 2011, vol. 1, pp. 125–130.

[1] F. Timm and E. Barth, “Accurate, fast, and robust centre localisation for images of semiconductor components,” 2011, vol. 7877, no. 0, pp. 787705–787705–10.

https://github.com/trishume/eyeLike

[1] D. E. King, “Dlib-ml: A Machine Learning Toolkit,” J. Mach. Learn. Res., vol. 10, pp. 1755–1758, Dec. 2009.

Corpora:

[1] F. Song, X. Tan, X. Liu, and S. Chen, “Eyes closeness detection from still images with multi-scale histograms of principal oriented gradients,” Pattern Recognit., vol. 47, no. 9, pp. 2825–2838, Sep. 2014.

[1] B. A. Smith, Q. Yin, S. K. Feiner, and S. K. Nayar, “Gaze locking: passive eye contact detection for human-object interaction,” in Proceedings of the 26th annual ACM symposium on User interface software and technology - UIST ’13, 2013, pp. 271–280.
