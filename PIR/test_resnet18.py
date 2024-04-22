from utils import TestBase

import paddle
from paddle.vision.models import resnet18
import paddle.profiler as profiler


class TestResNet18(TestBase):
     def __init__(self):
          super().__init__()
          self.prepare_net()
          self.prepare_data()
     def prepare_net(self):
          self.net = resnet18(pretrained=True)
     def prepare_data(self):
          self.batch_size = 32
          self.input_shape = [self.batch_size, 3, 224, 224]
          self.input = paddle.rand(self.input_shape)
     def profile(self, use_cinn):
          print("--[profile] profile %s" % ("cinn" if use_cinn else "nocinn"))
          net = self.to_eval(use_cinn)
          # 创建性能分析器 Profiler 对象，并配置参数。
          # 分析第 [15, 17）次迭代区间的性能数据。
          prof = profiler.Profiler(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
                              scheduler = (15, 17),
                              on_trace_ready = profiler.export_chrome_tracing('./profiler_data/resnet18/hygon', 'bs%s_%s' % (self.batch_size, 'withcinn' if use_cinn else "nocinn")),
                              timer_only = False)
          prof.start()
          for iteration in range(20):
               # 为作用域代码加"resent18 eval"的标签
               with profiler.RecordEvent(name="resent18 eval"):
                    output = net(self.input)
                    paddle.device.synchronize()
               prof.step()
          prof.stop()
          prof.summary(sorted_by=profiler.SortedKeys.GPUTotal,
                    op_detail=True,
                    thread_sep=False,
                    time_unit='ms')

if __name__ == '__main__':
     model = TestResNet18()
     model.check_cinn_ouput()
     model.benchmark(use_cinn=False)
     model.benchmark(use_cinn=True)
     model.profile(use_cinn=True)

# python3 test_resnet18.py &>> ./profiler_data/resnet18/hygon/bs32.log