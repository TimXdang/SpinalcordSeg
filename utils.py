import sys
import torch
from tqdm import tqdm as tqdm
from segmentation_models_pytorch.utils.meter import AverageValueMeter
import segmentation_models_pytorch.utils as smpu


# adaptions made to the classes from segmentation models
class Epoch:
    def __init__(self, model, loss, metrics, stage_name, device="cpu", verbose=True, unet2d=False):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

        self.unet2d = unet2d

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        old_loss = smpu.losses.DiceLoss()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        old_loss_meter = {'old_dice': AverageValueMeter()}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not (self.verbose),
        ) as iterator:
            for data in iterator:
                x = data['img']['data']
                y = data['label']['data']
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                old_dice = old_loss(y_pred, y).cpu().detach().numpy()
                old_loss_meter['old_dice'].add(old_dice)
                old_logs = {'old_dice': old_loss_meter['old_dice'].mean}
                logs.update(old_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, metrics, optimizer, device="cpu", verbose=True, unet2d=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
            unet2d=unet2d
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        if self.unet2d is True:
            batch_size = x.shape[0]
            o_shape = x.shape
            x = x.reshape(batch_size, 1, 160, -1)

            prediction = self.model.forward(x).reshape(o_shape)
        else:
            prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True, unet2d=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
            unet2d=unet2d
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            if self.unet2d is True:
                batch_size = x.shape[0]
                o_shape = x.shape
                x = x.reshape(batch_size, 1, 160, -1)

                prediction = self.model.forward(x).reshape(o_shape)
            else:
                prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction


class EarlyStopping:
    """
    Early stopping to stop training if no improvement.

    :ivar patience: patience (number of epochs without improvement until stop)
    :ivar min_delta: threshold by how much loss should improve at least
    :ivar counter: counts epochs without improvement
    :ivar early_stop: indicates whether training should be stopped
    """
    def __init__(self, patience: int = 5, min_delta: float = 0):

        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, new_loss, old_loss):
        if (new_loss - old_loss) > self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
