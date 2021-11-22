# downloading the files using gdown 
# https://github.com/wkentaro/gdown

import os
import gdown
import shutil

# file ids
links = {
    "1C3Kqqu9gDXNNXr5WDpmhGTr-UaQ2ckqQ": ["obj_detection.weights", "detector/YOLOv3/weight/"],
    "1VfRVvc-7EowI540S0_6FxJOMVg9XyXef": ["traj_arm.pth", "Trajectory/models/Individual/"],
    "1t_qok3BNNHN6EK_Uw3WiXrfGtLfQJpCs": ["traj_endeffector.pth", "Trajectory/models/Individual/"],
    "1SEZGUvLB2gfVwA-yGTBlF5k3YDiOzYWj": ["traj_probe.pth", "Trajectory/models/Individual/"],
    "1p8vo9rig9Q0i0WLVudQBgzkdjtJXcDiS": ["00013.pth", "Trajectory/models/Individual/eth_train/"],
    "1MlXnCSjD5yOfxnnJMkruE0rCgLCMZJlB": ["ckpt.t7", "deep_sort/deep/checkpoint/"],
    "13mVaJTsJ7rN-Bz5KtsS20dX1TZFAHLkU": ["test_video.mp4", "data/"]
}

# download the files and save to respective folder
cwd = os.getcwd()
for id in links:
  filename = links[id][0]
  filepath = links[id][1]
  destination = os.path.join(cwd, filepath)

  if not os.path.isdir(destination):
    os.mkdir(destination)

  url = f"https://drive.google.com/uc?id={id}"
  gdown.download(url, filename, quiet=False)
  shutil.move(os.path.join(cwd, filename), destination)
