import torch
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier():
    def __init__(self):
        self.net = None
        self.K = 0
        self.classifier = None

    def update(self, net, K, train_dataloader):
        self.net = net
        self.K = K
        # Inizialization of the classifier
        self.classifier = KNeighborsClassifier(n_neighbors = self.K)
        self.net = self.net.cuda()
        self.net.train(False)
        with torch.no_grad():
            for images, labels in train_dataloader:
                images = images.cuda()
                labels = labels.cuda()
                outputs, features = self.net(images, output = 'all')
                features = features.cpu()
                outputs = outputs.cpu()
                targets = torch.argmax(outputs, dim=1).cpu()
                # Fit the classifier on the train_dataloader (including examplars)
                self.classifier.fit(features, targets)
           
    def classify(self, images):
        preds = []
        self.net = self.net.cuda()
        self.net.train(False)
        with torch.no_grad():
            features = self.net(images, output = 'features')
            features = features.cpu()
            # Predictions of the classifiers
            preds = self.classifier.predict(features)
        return torch.Tensor(preds).cuda()



