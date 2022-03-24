# python3 total_onnx_every_frame_with_collision.py -m yolo_converted.onnx -i data/test_video.mp4 -o results/new_pipeline.mp4
# python3 onnx_traj_every_frame.py -m yolo_converted.onnx -i data/test_video.mp4 -o results/current_pipeline.mp4
# python3 previous_pipeline_with_collision.py -i data/test_video.mp4 --save_path results


python3 old_pipeline.py -i data/test_video.mp4 -q 1
python3 onnx_detector_pytorch_predictor.py -m yolo_converted.onnx -i data/test_video.mp4 -q 1
python3 pytorch_detector_onnx_trajectory.py -i data/test_video.mp4 -q 1
python3 new_pipeline.py -m yolo_converted.onnx -i data/test_video.mp4 -q 1
