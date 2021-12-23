import os
import torch
import argparse
from Trajectory import individual_TF

def exportOnnxUtil(model, input_torch, modelname):
    torch.onnx.export(model,
                  input_torch,
                  modelname,
                  export_params=True,
                  opset_version=10,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

def exportOnnx(output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_torch = torch.rand(1, 7, 2)

    # create directory if doesn't exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # end effector onnx model
    traj_endeffector = individual_TF.IndividualTF(
        2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]).to(device)
    traj_endeffector.load_state_dict(torch.load(
        f'data/traj_endeffector.pth', map_location=torch.device('cpu')))
    traj_endeffector.eval()
    output_name = os.path.join(output_dir, "endeffector.onnx")
    exportOnnxUtil(traj_endeffector, input_torch, output_name)

    # problance onnx model
    traj_problance = individual_TF.IndividualTF(
        2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]).to(device)
    traj_problance.load_state_dict(torch.load(
        f'data/traj_problance.pth', map_location=torch.device('cpu')))
    traj_problance.eval()
    output_name = os.path.join(output_dir, "problance.onnx")
    exportOnnxUtil(traj_problance, input_torch, output_name)

    # probe onnx model
    traj_probe = individual_TF.IndividualTF(
        2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]).to(device)
    traj_probe.load_state_dict(torch.load(
        f'data/traj_probe.pth', map_location=torch.device('cpu')))
    traj_probe.eval()
    output_name = os.path.join(output_dir, "probe.onnx")
    exportOnnxUtil(traj_probe, input_torch, output_name)

    # person onnx model
    traj_person = individual_TF.IndividualTF(
        2, 3, 3, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, mean=[0, 0], std=[0, 0]).to(device)
    traj_person.load_state_dict(torch.load(
        f'data/traj_person.pth', map_location=torch.device('cpu')))
    traj_person.eval()
    output_name = os.path.join(output_dir, "person.onnx")
    exportOnnxUtil(traj_person, input_torch, output_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="helper script to export models to convert to onnx format")
    parser.add_argument('-o', '--output_dir', type=str, help="Output directory to create onnx models")

    args = parser.parse_args()

    exportOnnx(args.output)