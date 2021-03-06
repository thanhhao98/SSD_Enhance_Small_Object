import torch
from torch import nn
from torchvision.ops import nms
from models.utils import (
    Bottleneck,
    PyramidFeatures,
    RegressionModel,
    ClassificationModel,
    BBoxTransform,
    ClipBoxes,
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
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()

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

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

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
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(
                transformed_anchors,
                img_batch
            )
            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates

