import paddle
from paddle.vision.models import resnet18

import os

# model
model = resnet18(pretrained=True)
x = paddle.rand([1, 3, 224, 224])
# paddle inference
model.eval()
print(model(x)[0][:10])
# paddle with cinn inference
'''
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_prim_all'] = 'true'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_cinn'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
# 设置cinn支持的算子
paddle.set_flags({"FLAGS_allow_cinn_ops": "abs;"})


build_strategy = paddle.static.BuildStrategy()
build_strategy.build_cinn_pass = True
static_model = paddle.jit.to_static(
    model,
    build_strategy=build_strategy,
    full_graph=True,
)
static_model.eval()
out = static_model(x)
print(out[0][:10])
'''