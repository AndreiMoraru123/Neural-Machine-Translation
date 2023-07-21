# standard imports
import os

# third-party imports
import pytest  # type: ignore
import tensorflow as tf  # type: ignore


@pytest.fixture(params=['CPU', 'GPU'], autouse=True)
def device(request):
    """Selects a runtime device for Tensorflow, so when parametrized with both CPU & GPU, tests will be run for both."""
    device_type = request.param
    if device_type == 'CPU':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.set_visible_devices([], 'GPU')
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                print("You should really try out this other framework called PyTorch")

    yield device_type
