# FlowPose6D: Flow driven 6D Object Pose Refinement
This repository contains the source code for our paper:

FlowPose6D: Flow driven 6D Object Pose Refinement 3DV
Jonas Frey and Kenneth Blomqvist 

TODO:
- Train Optical Flow using different parameters

### Experiments and Metrices to Report:
#### 1
- Correlation: EPE - Iterations FLow Estimation
- Correlation: EPE -> Pose Error
- Correlation: Segmentation -> PoseError

#### 2
- Timing of network.
- Tracking results with PoseCNN init.
- PoseCNN init results.

#### 3
- Using P3P and Ransac to verify geometric consistency of flow estimate
- PoseRejection Module
- Evaluate increase in performance



### DONE:
- Create inital Poses using PoseCNN
- Visu Segmentation and Flow
- Inference Helper to evaluate different settings