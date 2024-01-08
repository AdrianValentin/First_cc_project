#from tests import _PATH_DATA
from first_cc_project.train_model2 import train2
import pytest

import os.path


# this will be found and executed by pytest
def test_something():
    #train(lr = 0.01, batch_size = 4, num_epochs = 2)
    #train2(lr = 0.01, batch_size = 4, num_epochs = 2)
    
    assert True
    
def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Too high a learning rate'):
        train2(lr = 1, batch_size = 4, num_epochs = 2)        
