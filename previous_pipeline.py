import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
from halo import Halo
from detector import build_detector
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from Trajectory import individual_TF
from Trajectory.transformer.batch import subsequent_mask
from tool.utils import load_class_names

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class VideoTracker(object):
    def __init__(self, cfg , args , video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")
        self.spinner = Halo(text="loading frames", spinner='dots')
        self.class_names = load_class_names("data/names")
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        
        self.traj_ped = individual_TF.IndividualTF(2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0,0], std=[0,0]).to(device)
        self.traj_ped.load_state_dict(torch.load("data/traj_person.pth", map_location=torch.device('cpu')))
        self.traj_ped.eval()
        self.traj_endeffector = individual_TF.IndividualTF(2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0,0], std=[0,0]).to(device)
        self.traj_endeffector.load_state_dict(torch.load(f'data/traj_endeffector.pth', map_location=torch.device('cpu')))
        self.traj_endeffector.eval()
        self.traj_arm = individual_TF.IndividualTF(2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]).to(device)
        self.traj_arm.load_state_dict(torch.load(f'data/traj_problance.pth', map_location=torch.device('cpu')))
        self.traj_arm.eval()
        self.traj_probe = individual_TF.IndividualTF(2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]).to(device)
        self.traj_probe.load_state_dict(torch.load(f'data/traj_probe.pth', map_location=torch.device('cpu')))
        self.traj_probe.eval()
        self.class_names = self.detector.class_names
        self.Q = { }
        self.previous_prediction_fps = -1

    
    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.mp4")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))
        self.spinner.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.spinner.stop()
        if self.vdo:
            self.vdo.release()
        cv2.destroyAllWindows()
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        mean_end_effector = torch.tensor((-2.6612e-05, -7.8652e-05))
        std_end_effector = torch.tensor((0.0025, 0.0042))
        mean_arm = torch.tensor([-1.3265e-05, -6.5026e-06])
        std_arm = torch.tensor([0.0030, 0.0185])
        mean_probe = torch.tensor([-5.1165e-05, -7.1806e-05])
        std_probe = torch.tensor([0.0038, 0.0185])
        mean_ped = torch.tensor([0.0001, 0.0001])
        std_ped = torch.tensor([0.0001, 0.0001])

        while self.vdo.grab() :
            idx_frame += 1
            if self.args.frame_count and self.args.frame_count < idx_frame:
                break
            if idx_frame % self.args.frame_interval:
                continue
            start = time.time()
            _, ori_im = self.vdo.retrieve()

            detection_start = time.time()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            height, width = ori_im.shape[:2]
            bbox_xywh , cls_conf, cls_ids = self.detector(im)
            
            for i in range(5):
                mask = cls_ids == i
                t_cls_conf = cls_conf[mask]
                t_bbox_xywh = bbox_xywh[mask]
                if t_cls_conf.size > 0:
                    cx = t_bbox_xywh[np.argmax(t_cls_conf)][0]
                    cy = t_bbox_xywh[np.argmax(t_cls_conf)][1]
                    w = t_bbox_xywh[np.argmax(t_cls_conf)][2]
                    h = t_bbox_xywh[np.argmax(t_cls_conf)][3]
                    x1 = int(cx - (w/2))
                    y1 = int(cy - (h/2))
                    x2 = int(cx + (w/2))
                    y2 = int(cy + (h/2))
                    pt = [t_bbox_xywh[np.argmax(t_cls_conf)][0] / width, t_bbox_xywh[np.argmax(t_cls_conf)][1] / height]
                    ori_im = cv2.rectangle(ori_im, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    ori_im = cv2.putText(ori_im, self.class_names[i], (x1, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
                    t_id = i
                    if t_id in self.Q:
                        self.Q[t_id][0].append(pt)
                    else:
                        self.Q[t_id] = [[pt]]
            
            detection_end = time.time()
            # select person class
            mask = cls_ids == 3
            bbox_xywh = bbox_xywh[mask]
            cls_conf = cls_conf[mask]

            prediction_start = time.time()
            prediction_count = 0
            for i in self.Q:
                if (len(self.Q[i][0])) == 8:
                    prediction_count += 1
                    Q_np = np.array(self.Q[i], dtype=np.float32)
                    Q_d = Q_np[:, 1:, 0:2] - Q_np[:, :-1, 0:2]
                    pr = []
                    inp = torch.from_numpy(Q_d)
                    #print(i)
                    #print(inp)
                    if i == 0:
                        inp = (inp.to(device) - mean_end_effector.to(device)) / std_end_effector.to(device)
                    elif i == 1:
                        inp = (inp.to(device) - mean_arm.to(device)) / std_arm.to(device)
                    elif i == 2:
                        inp = (inp.to(device) - mean_probe.to(device)) / std_probe.to(device)
                    else:
                        inp = (inp.to(device) - mean_ped.to(device)) / std_ped.to(device)
                    src_att = torch.ones((inp.shape[0], 1, inp.shape[1])).to(device)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                        device)
                    dec_inp = start_of_seq
                    for itr in range(12):
                        trg_att = subsequent_mask(dec_inp.shape[1]).repeat(dec_inp.shape[0], 1, 1).to(device)
                        if i == 0:
                            out = self.traj_endeffector(inp, dec_inp, src_att, trg_att)
                        elif i == 1:
                            out = self.traj_arm(inp, dec_inp, src_att, trg_att)
                        elif i == 2:
                            out = self.traj_probe(inp, dec_inp, src_att, trg_att)
                        else:
                            out = self.traj_ped(inp, dec_inp, src_att, trg_att)
                        dec_inp = torch.cat((dec_inp, out[:, -1:, :]), 1)
                    if i == 0:
                        preds_tr_b = (dec_inp[:, 1:, 0:2] * std_end_effector.to(device) + mean_end_effector.to(device)).detach().cpu().numpy().cumsum(1)+Q_np[:, -1:, 0:2]
                    elif i == 1:
                        preds_tr_b = (dec_inp[:, 1:, 0:2] * std_arm.to(device) + mean_arm.to(device)).detach().cpu().numpy().cumsum(1) + Q_np[:, -1:, 0:2]
                    elif i == 2:
                        preds_tr_b = (dec_inp[:, 1:, 0:2] * std_probe.to(device) + mean_probe.to(device)).detach().cpu().numpy().cumsum(1) + Q_np[:, -1:, 0:2]
                    else:
                        preds_tr_b = (dec_inp[:, 1:, 0:2] * std_ped.to(device) + mean_ped.to(device)).detach().cpu().numpy().cumsum(1) + Q_np[:, -1:, 0:2]
                    pr.append(preds_tr_b)
                    pr = np.concatenate(pr, 0)
                    self.Q[i][0].pop(0)
                    co = (0, 255, 0)  # green
                    cp = (0, 0, 255)  # red
                    #print(pr)
                    for j in range(11):
                        pp1 = (int(pr[0, j, 0]*width), int(pr[0, j, 1]*height))
                        pp2 = (int(pr[0, j+1, 0] * width), int(pr[0, j+1, 1] * height))
                        #ori_im = cv2.circle(ori_im, pp, 3, cp, -1)
                        ori_im = cv2.line(ori_im,pp1,pp2,cp,2)
                    for j in range(7):
                        op1 = (int(Q_np[0, j, 0]*width), int(Q_np[0, j, 1]*height))
                        op2 = (int(Q_np[0, j+1, 0] * width), int(Q_np[0, j+1, 1] * height))
                        #ori_im = cv2.circle(ori_im, op, 3, co, -1)
                        ori_im = cv2.line(ori_im, op1, op2, co, 2)
            

            prediction_end = time.time()
            end = time.time()

            detection_fps = round(1 / (detection_end - detection_start), 2)
            
            prediction_time = prediction_end - prediction_start
            if prediction_count == 0:
                prediction_fps = self.previous_prediction_fps
            else:
                prediction_fps = round(1 / (prediction_time / prediction_count), 2)
                self.previous_prediction_fps = prediction_fps
            total_fps = round(1 / (end - start), 2)
            self.spinner.text = f"[Frame {idx_frame}] [FPS] Detection: {detection_fps} Prediction: {self.previous_prediction_fps} Total: {total_fps}"
            ori_im = cv2.putText(ori_im, f"[FPS] Detection: {detection_fps} Prediction: {self.previous_prediction_fps} Total: {total_fps}", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            # cv2.imshow("result", ori_im)
            # cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to the input video")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_false")
    parser.add_argument("--frame_interval", type=int, default=10)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    parser.add_argument("-f", "--frame_count", type=int, default=None, help="Number of frames to run the script")    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.input) as vdo_trk:
        vdo_trk.run()

