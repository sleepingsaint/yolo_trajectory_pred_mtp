# Transformers ONNX

As we converted the YOLO models to ONNX format for ONNX Runtime with TensorRT execution provider, we also convert Transformers to ONNX specification.

{% hint style="info" %}
If you want to read more about ONNX check [ONNX](../notes/onnx.md) section in notes section
{% endhint %}

## Conversion

Since the models are already trained in PyTorch, we can easily converted pyTorch models to ONNX specification using PyTorch ONNX utilities.&#x20;

Below is the example of converting a trained model to ONNX.

```python
x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
torch_out = torch_model(x)
python
# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
```

{% hint style="info" %}
In the main pipeline, all the pretrained ONNX models will be downloaded automatically with download.py script
{% endhint %}
