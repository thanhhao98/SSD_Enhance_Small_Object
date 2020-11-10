import torch
from models import ResNet

model = ResNet(
    [3, 8, 36, 3],
    in_channels=3,

).cuda()
model.training = False
model.eval()
x = torch.rand(1, 3, 512, 512).cuda()
finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates = model(x)
print(finalScores.size())
print(finalAnchorBoxesIndexes.size())
print(finalAnchorBoxesCoordinates.size())
print(finalScores[:10])
print(finalAnchorBoxesIndexes[:10])
print(finalAnchorBoxesCoordinates[:10])

