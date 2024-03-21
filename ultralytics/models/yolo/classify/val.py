# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data import AgeGenderClassificationDataset, ClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.metrics import ClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class ClassificationValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'classify'
        self.metrics = ClassifyMetrics()

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        return ('%22s' + '%11s' * 2) % ('classes', 'top1_acc', 'top5_acc')

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, task='classify')
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = batch['img'].half() if self.args.half else batch['img'].float()
        batch['cls'] = batch['cls'].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        n5 = min(len(self.model.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5])
        self.targets.append(batch['cls'])

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir,
                                           names=self.names.values(),
                                           normalize=normalize,
                                           on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        return ClassificationDataset(root=img_path, args=self.args, augment=False)

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""
        pf = '%22s' + '%11.3g' * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ('all', self.metrics.top1, self.metrics.top5))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            images=batch['img'],
            batch_idx=torch.arange(len(batch['img'])),
            cls=batch['cls'].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f'val_batch{ni}_labels.jpg',
            names=self.names,
            on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=torch.argmax(preds, dim=1),
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)  # pred


class AgeGenderClassificationValidator(BaseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'classify_age_gender'
        self.metrics_age = ClassifyMetrics("age")
        self.metrics_gender = ClassifyMetrics("gender")

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        return ('%22s' + '%11s' * 4) % ('classes', 'precision', 'recall', 'accuracy', 'F1')

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and prec, rec, acc."""
        self.names_age = {0: "Child", 1: "Teen", 2: "Adult", 3: "Senior", 4: "Unknown"}
        self.names_gender = {0: "Male", 1: "Female", 2: "Unknown"}
        self.nc_age = len(self.names_age)
        self.nc_gender = len(self.names_gender)
        self.confusion_matrix_age = ConfusionMatrix(nc=self.nc_age, class_category='age', task='classify')
        self.confusion_matrix_gender = ConfusionMatrix(nc=self.nc_gender, class_category='gender', task='classify')
        self.pred_age = []
        self.pred_gender = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = batch['img'].half() if self.args.half else batch['img'].float()
        batch['cls'] = batch['cls'].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        n5_age = min(len(self.names_age), 5)
        n5_gender = min(len(self.names_gender), 5)

        self.pred_age.append(preds[0].argsort(1, descending=True)[:, :n5_age])
        self.pred_gender.append(preds[1].argsort(1, descending=True)[:, :n5_gender])
        self.targets.append(batch['cls'])

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        # Age
        targets_age = [batch_target[:, 0] for batch_target in self.targets]
        self.confusion_matrix_age.process_cls_preds(self.pred_age, targets_age)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix_age.plot(save_dir=self.save_dir,
                                               names=self.names_age.values(),
                                               normalize=normalize,
                                               on_plot=self.on_plot)
        self.metrics_age.speed = self.speed
        self.metrics_age.confusion_matrix = self.confusion_matrix_age

        # Gender
        targets_gender = [batch_target[:, 1] for batch_target in self.targets]
        self.confusion_matrix_gender.process_cls_preds(self.pred_gender, targets_gender)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix_gender.plot(save_dir=self.save_dir,
                                           names=self.names_gender.values(),
                                           normalize=normalize,
                                           on_plot=self.on_plot)
        self.metrics_gender.speed = self.speed
        self.metrics_gender.confusion_matrix = self.confusion_matrix_gender

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        targets_age = [batch_target[:, 0] for batch_target in self.targets]
        targets_gender = [batch_target[:, 1] for batch_target in self.targets]
        self.metrics_age.process(targets_age, self.pred_age)
        self.metrics_gender.process(targets_gender, self.pred_gender)
        return [self.metrics_age.results_dict, self.metrics_gender.results_dict]

    def build_dataset(self, img_path):
        return AgeGenderClassificationDataset(root=img_path, args=self.args, augment=False)

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""
        # Age
        pf = '%22s' + '%11.3g' * len(self.metrics_age.keys)  # print format
        LOGGER.info(pf % ('Age',
                          self.metrics_age.precision,
                          self.metrics_age.recall,
                          self.metrics_age.accuracy,
                          self.metrics_age.fitness))

        # Gender
        pf = '%22s' + '%11.3g' * len(self.metrics_gender.keys)  # print format
        LOGGER.info(pf % ('Gender',
                          self.metrics_gender.precision,
                          self.metrics_gender.recall,
                          self.metrics_gender.accuracy,
                          self.metrics_gender.fitness))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            images=batch['img'],
            batch_idx=torch.arange(len(batch['img'])),
            cls=batch['cls'][:, 0].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f'val_batch{ni}_age_labels.jpg',
            names=self.names_age,
            on_plot=self.on_plot)

        plot_images(
            images=batch['img'],
            batch_idx=torch.arange(len(batch['img'])),
            cls=batch['cls'][:, 1].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f'val_batch{ni}_gender_labels.jpg',
            names=self.names_gender,
            on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=torch.argmax(preds[0], dim=1),
                    fname=self.save_dir / f'val_batch{ni}_age_pred.jpg',
                    names=self.names_age,
                    on_plot=self.on_plot)  # pred

        plot_images(batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=torch.argmax(preds[1], dim=1),
                    fname=self.save_dir / f'val_batch{ni}_gender_pred.jpg',
                    names=self.names_gender,
                    on_plot=self.on_plot)  # pred


def val(cfg=DEFAULT_CFG, use_python=False):
    """Validate YOLO model using custom data."""
    model = cfg.model or 'yolov8n-cls.pt'  # or "resnet18"
    data = cfg.data or 'mnist160'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = ClassificationValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()
