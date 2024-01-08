from tests import _PATH_DATA
from first_cc_project.train_model import processed_mnist
import numpy as np
import torch
import pytest
import os.path
#@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")


def test_data():
    train_data, test_data = processed_mnist()

    assert len(train_data) == 30000 
    assert len(test_data) == 6000
    #assert that each datapoint has shape [1,28,28] or [784] depending on how you choose to format
    #print(train_data[0][0].shape)
    #print(train_data[0].shape)
    assert all(train_data[image][0].shape == torch.Size([1, 28, 28]) for image in range(len(train_data))), "train data shape is wrong"
    assert all(test_data[image][0].shape == torch.Size([1, 28, 28]) for image in range(len(test_data))), "test data shape is wrong"

    #assert that all labels are represented
    assert all([digit in [train_data[label][1] for label in range(len(train_data))] for digit in range(10)]), "train data labels are wrong"
    assert all([digit in [test_data[label][1] for label in range(len(test_data))] for digit in range(10)]), "test data labels are wrong"


