# Gaze-Tracker
An eye localization and gaze tracking system for a single low cost camera

## Overview
This Matlab software is the implementation of the Skodras et al. work. The software is able to localize the eyes exploiting the color information and the radial simmetry (Fast Radial Symmetry transform by Loy et al.). After eye localization, the center of a patch is used as an anchor point. The patch is tracked frame by frame using the Kanade–Lucas–Tomasi (KLT) feature tracker algorithm. Finally, a linear regression model is calibrated using a 10-points pattern and the model is succesively used to estimate the gaze position on the screen. 

For detailed information refer to project report.
