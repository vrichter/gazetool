#!/bin/bash
prefix="/homes/lschilli/src/dlibgazer/install"
exec "$prefix/bin/dlibgazer" -m "$prefix/share/dlibgazer/shape_predictor_68_face_landmarks.dat" --estimate-gaze "$prefix/share/dlibgazer/gaze_est_deg.dat" --estimate-verticalgaze "$prefix/share/dlibgazer/vertgaze_est_deg.dat" --estimate-lid "$prefix/share/dlibgazer/lid_est.dat" "$@"
