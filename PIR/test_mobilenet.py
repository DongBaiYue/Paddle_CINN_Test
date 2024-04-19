import paddle
from paddle.vision.models import mobilenet_v2

#model = mobilenet_v2()
model = mobilenet_v2(pretrained=True)