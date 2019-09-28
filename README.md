# Gaze-Tracker
An eye localization and gaze tracking system for a single low cost camera

## Overview
Gaze-Tracker is a Matlab program that localizes the eye centers and plots the estimated gaze on the screen using a Feature-Based approach. The program is thought to work in real-time with a single low-cost camera, such those you can ﬁnd on PCs, tablets, and smartphones. Gaze-Tracker is the implementation of the paper by Skodras et al. “On visual gaze tracking based on a single low cost camera”. The software can localize the eyes exploiting the color information and the radial symmetry (Fast Radial Symmetry transform by Loy et al.). After eye localization, the center of a patch is used as an anchor point. The patch is tracked frame by frame using the Kanade–Lucas–Tomasi (KLT) feature tracker algorithm. Finally, a linear regression model is calibrated using a 10-points pattern and successively used to estimate the gaze position on the screen.

For detailed information, refer to the project [report](doc/Gaze-Tracker.pdf).

