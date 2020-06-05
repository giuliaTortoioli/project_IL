from torch.utils.data import DataLoader

class IncrementalDualMemoryClassifier():
    def __init__(self):
        self.net = None
        self.current_step = None
        self.mean_train_scores = {}
        self.mean_examplars_scores = {}
        self.confidences = {}

    def update(step, net, train_dataloader):
        self.net = net
        self.current_step = step

        # initialize the scores for the new classes
        num_new_classes = len(train_dataloader.dataset.stored_labels)
        num_old_classes = len(self.mean_train_scores)
        self.mean_examplars_scores = {k: 0 for k in range(num_old_classes)}
        for i in range(num_new_classes):
            self.mean_train_scores[num_old_classes + i] = (step, 0)
        num_images = [0 for _ in range(num_new_classes + num_old_classes)]

        self.net = self.net.cuda()
        self.net.train(False)
        with torch.no_grad():
            for images, labels in train_dataloader:
                images = images.cuda()
                labels = labels.cuda()
                scores = net(images)
                for score, label in zip(scores, labels):
                    # exclude examplars from old classes for the updating of the scores
                    score = score.cpu()
                    self.confidences[step] += torch.max(score).item()
                    if label >= num_old_classes:
                        self.mean_train_scores[label][1] += score[label]
                    else:
                        self.mean_examplars_scores[label] += score[label]
                    num_images[label] += 1

        self.confidences[step] /= len(train_dataloader.dataset)
        # dividing for the number of images per label
        for i in range(num_old_classes):
            self.mean_examplars_scores[i] /= num_images[i]
        for i in range(num_new_classes):
            self.mean_train_scores[num_old_classes + i][1] /= num_images[num_old_classes + i]

    def rectify(score, num_old_classes):
        for i in range(num_old_classes):
            old_step = self.mean_train_scores[i][0]
            mu_p_class = self.mean_train_scores[i][1]
            mu_n_class = self.mean_examplars_scores[i]
            conf_p = self.confidences[old_step]
            conf_n = self.confidences[self.current_step]
            score[i] = score[i] * (mu_p_class/mu_n_class) * (conf_n/conf_p)
        return score

    def classify(images):
        num_old_classes = len(self.mean_examplars_scores)
        preds = []
        with torch.no_grad():
            self.net = self.net.cuda()
            self.net.train(False)
            scores = net(images)

            for score in scores:
                score = score.cpu()
                pred = torch.argmax(score)
                if pred >= num_old_classes:
                    # rectify scores for old classes
                    score = rectify(score, num_old_classes)
                    pred = torch.argmax(score).item()
                preds.append(pred)

        return torch.Tensor(preds).cuda()
