import sys
from os.path import dirname
sys.path.append(dirname(dirname(__file__)))

from utils import TestBase

import paddle
from paddle.nn import Transformer

paddle.set_flags({"FLAGS_deny_cinn_ops": "conv2d;conv2d_grad;depthwise_conv2d;uniform_random"})

class TestTransformer(TestBase):
     def __init__(self):
          super().__init__()
          paddle.seed(2024)
          self.prepare_net()
          self.prepare_data()
     def prepare_net(self):
          self.nocinn_net = Transformer(128, 2, 4, 4, 512)
     def prepare_data(self):
        # src: [batch_size, tgt_len, d_model]
        enc_input = paddle.rand((2, 4, 128))
        # tgt: [batch_size, src_len, d_model]
        dec_input = paddle.rand((2, 6, 128))
        # src_mask: [batch_size, n_head, src_len, src_len]
        enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
        # tgt_mask: [batch_size, n_head, tgt_len, tgt_len]
        dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
        # memory_mask: [batch_size, n_head, tgt_len, src_len]
        cross_attn_mask = paddle.rand((2, 2, 6, 4))
        self.input = [enc_input, dec_input, enc_self_attn_mask, dec_self_attn_mask, cross_attn_mask]

if __name__ == '__main__':
     model = TestTransformer()
     model.check_cinn_ouput()