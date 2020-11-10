import torch
from torch import nn
from models.utils import (
    Bottleneck,
    PyramidFeatures,
    RegressionModel,
    ClassificationModel
)
from models.anchors import (
    Anchors
)


class ResNet(nn.Module):
    def __init__(
            self,
            layers,
            in_channels=3,
            num_classes=80,
            fpn_feature_size=256,
            feature_size=256,
            planes=[64, 128, 256, 512]
    ):
        self.in_planes = planes[0]
        self.planes = planes
        self.fpn_feature_size = fpn_feature_size
        self.feature_size = feature_size
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            self.in_planes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.layer1 = self._make_layer(self.planes[0], layers[0])
        self.layer2 = self._make_layer(
            self.planes[1],
            layers[1],
            stride=2
        )
        self.layer3 = self._make_layer(
            self.planes[2],
            layers[2],
            stride=2
        )
        self.layer4 = self._make_layer(
            self.planes[3],
            layers[3],
            stride=2
        )
        fpn_sizes = [
            self.layer2[layers[1] - 1].conv3.out_channels,
            self.layer3[layers[2] - 1].conv3.out_channels,
            self.layer4[layers[3] - 1].conv3.out_channels
        ]
        self.fpn = PyramidFeatures(
            fpn_sizes[0],
            fpn_sizes[1],
            fpn_sizes[2],
            feature_size=self.fpn_feature_size
        )
        self.regressionModel = RegressionModel(
            fpn_feature_size=self.fpn_feature_size,
            feature_size=self.feature_size
        )
        self.classificationModel = ClassificationModel(
            feature_size=self.feature_size,
            num_classes=num_classes
        )
        self.anchors = Anchors()

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        expansion = Bottleneck.expansion
        if stride != 1 or self.in_planes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes,
                    planes * expansion,
                    kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * expansion),
            )
        layers = [Bottleneck(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        regression = torch.cat(
            [self.regressionModel(feature) for feature in features],
            dim=1
        )
        classification = torch.cat(
            [self.classificationModel(feature) for feature in features],
            dim=1
        )
        anchors = self.anchors(img_batch)
        if self.training:
            pass
        else:
            pass
        return features, regression, classification
