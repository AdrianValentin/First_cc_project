from first_cc_project.models.model_day1 import myawesomemodel
import torch
import numpy as np 
def test_model():
    import torch

    # Load the saved model
    model = torch.load('models/model.pt')
    model.eval()  # Set the model to evaluation mode

    # Assuming your input is a numpy array, convert it to a PyTorch tensor
    # For example, let's create a dummy input tensor
    input_tensor = torch.randn(1, 28, 28)

    # If your model expects a different input shape, adjust the tensor accordingly
    # For example, if your model expects a batch size and a channel dimension, you might need to reshape
    input_tensor = input_tensor.unsqueeze(0)  # Adds a batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)

    assert output[0].shape == torch.Size([10])
    # output is your model's prediction
