import paddle
import paddle.profiler as profiler
import os
import time
import numpy as np

def set_flags(use_cinn):
    if use_cinn:
        os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
        os.environ['FLAGS_group_schedule_tiling_first'] = '1'
        os.environ['FLAGS_prim_all'] = 'true'
        os.environ['FLAGS_prim_enable_dynamic'] = 'true'  
        os.environ['FLAGS_print_ir'] = '1'
        os.environ['FLAGS_enable_pir_api'] = '1'
        os.environ['FLAGS_use_cinn'] = '1'
        os.environ['FLAGS_cinn_bucket_compile'] = '1'
        os.environ['FLAGS_cinn_new_cluster_op_method'] = '1'
    else:
        os.environ['FLAGS_use_cinn'] = '0'

def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )

def benchmark(net, input, repeat=50, warmup=10):
    # warm up
    for i in range(warmup):
        net(input)
        paddle.device.synchronize()
    # time
    t = []
    for i in range(repeat):
        t1 = time.time()
        output = net(input)
        paddle.device.synchronize()
        t2 = time.time()
        t.append((t2 - t1)*1000)
    print("--[benchmark] Run for %d times, the average latency is:%f ms" % (repeat, np.mean(t)))

class TestBase:
    def __init__(self):
        self.net = None
        self.input = None
        self.cinn_net = None
    def to_eval(self, use_cinn):
        set_flags(use_cinn)
        if use_cinn:
            if not self.cinn_net:
                self.cinn_net = apply_to_static(self.net, use_cinn)
                self.prepare_net()
            net = self.cinn_net
        else:
            net = self.net
        net.eval()
        return net
    def eval(self, use_cinn):
        net = self.to_eval(use_cinn)
        out = net(self.input)
        return out
    def check_cinn_ouput(self):
        cinn_out = self.eval(use_cinn=True)
        dy_out = self.eval(use_cinn=False)
        np.testing.assert_allclose(
            cinn_out.numpy(), dy_out.numpy(), atol=1e-5, rtol=1e-5
        )
        print("--[check_cinn_ouput] cinn result right.")
    def benchmark(self, use_cinn):
        print("--[benchmark] benchmark %s" % ("cinn" if use_cinn else "nocinn"))
        net = self.to_eval(use_cinn)
        benchmark(net, self.input)
