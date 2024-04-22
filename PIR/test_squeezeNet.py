import paddle
from paddle.vision.models import squeezenet1_1


model = squeezenet1_1(pretrained=True)