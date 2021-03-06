import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Nearest mean of examplars classifier
# [net]: the feature extractor
# [examplars]: the examplars of the old classes
# [train_dataset]: the SubCIFAR object containing the data of the new classes
# [transform]: transform image into tensor and normalize
class NearestMeanOfExamplarsClassifier():
  def __init__(self, net, examplars, train_dataset, transform):
    with torch.no_grad():
      self.net = net
      self.net.train(False)
      self.means = []
      for i in range(len(examplars) + len(train_dataset.stored_labels)):
        # for the new classes, consider as examplar set all the available training
        if i >= len(examplars):
          mapped_label = train_dataset.all_labels[i]
          label_mask = (train_dataset.dataFrame["label"] == mapped_label)
          examplar_set = train_dataset.dataFrame[label_mask]["image"].values
        else:
          examplar_set = examplars[i][:,0]
          
        features_mean = torch.zeros((net.fc.in_features,))

        for j in range(len(examplar_set)):
          tensor = transform(examplar_set[j]).unsqueeze(0).cuda()

          features = self.net(tensor, output = 'features')
          features = features.squeeze(0).cpu()
          features = features / torch.norm(features, p = 2)
          
          features_mean += features
        features_mean /= len(examplar_set)
        features_mean = features_mean / torch.norm(features_mean, p = 2)
        self.means.append(features_mean)

      self.means = torch.stack(self.means)
      
  # predict the labels for the batch [input_images]
  # according to the nearest mean criterion 
  def classify(self, input_images):
    with torch.no_grad():
      self.net.train(False)
      features = self.net(input_images, output = 'features')
      features = F.normalize(features, p = 2)
      preds = []
      for feature in features:
        distances = torch.pow(self.means.cuda() - feature, 2).sum(-1)
        preds.append(distances.argmin().item())
      return torch.Tensor(preds).cuda()
