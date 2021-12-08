# downloading the files using gdown 
# https://github.com/wkentaro/gdown

import os
import gdown
import shutil

# file ids
links = {
    "1VfRVvc-7EowI540S0_6FxJOMVg9XyXef": ["traj_arm.pth", "data/"],
    "1t_qok3BNNHN6EK_Uw3WiXrfGtLfQJpCs": ["traj_endeffector.pth", "data/"],
    "1SEZGUvLB2gfVwA-yGTBlF5k3YDiOzYWj": ["traj_probe.pth", "data/"],
    "1p8vo9rig9Q0i0WLVudQBgzkdjtJXcDiS": ["00013.pth", "data/"],
    "13mVaJTsJ7rN-Bz5KtsS20dX1TZFAHLkU": ["test_video.mp4", "data/"],
    "1P2pV7CTRZbBoqv3i3np2WwU9wCVyHIxG": ["yolov3TATA608_final.weights", "data/"],
    "154S6WyZqLeF3G3gPTMibGnLnVLcpmEwm": ["test_image.jpg", "data/"],
    "1Kf6NuBeEbeXUxau3uQZVR02ry422wxv2DdZe6G0B2c4": ["yolov3TATA608.cfg", "data/"]
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
