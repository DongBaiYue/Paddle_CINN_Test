import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

from utils import apply_to_static

import paddle
from paddle import nn
import unittest
import numpy as np

class AbsNet(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = paddle.abs(x)
        return out

class TestAbs(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def prepare_data(self):
        self.shape = [4, 4, 4096]
        self.x = paddle.randn(self.shape, dtype="float32")
        self.x.stop_gradient = False # or True

    def eval(self, use_cinn):
        net = AbsNet()
        net = apply_to_static(net, use_cinn)
        net.eval()
        out = net(self.x)
        return out

    def test_eval(self):
        dy_out = self.eval(use_cinn=False)
        cinn_out = self.eval(use_cinn=True)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-6, rtol=1e-6
        )

if __name__ == '__main__':
    unittest.main()