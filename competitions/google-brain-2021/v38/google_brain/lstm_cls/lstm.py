import gc
import os
import random
from math import log
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from pandas import DataFrame, Series
from teads.models.metrics.metrics import Metrics
from teads.models.model_config import ModelConfig
from teads.models.model_wrapper import ModelWrapper
from teads.models.types import FitResult
from teads.util.average_meter import AverageMeter
from teads.util.logger import Logger
from teads.util.timer import Timer
from torch.nn.modules.normalization import LayerNorm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ExponentialLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup


class GoogleBrainLSTMConfig(ModelConfig):
    def __init__(
        self,
        save_dir: str,
        save_file_name: str,
        model_file_type: str,
        seed: int,
        epoch: int,
        batch_size: int,
        categorical_features: List[str],
        continuous_features: List[str],
        unique_targets: List[float],
        target_dict_inv: Dict,
        target_dict: Dict,
        device: str,
        use_amp: bool,
        num_workers: int,
        print_freq: int = 100,
        is_debug: bool = False,
    ) -> None:
        super().__init__(save_dir=save_dir, save_file_name=save_file_name, model_file_type=model_file_type)

        self.seed = seed
        self.epoch = epoch
        self.batch_size = batch_size
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.unique_targets = unique_targets
        self.target_dict = target_dict
        self.target_dict_inv = target_dict_inv
        self.device = device
        self.use_amp = use_amp

        self.num_workers = num_workers
        self.print_freq = print_freq

        self.is_debug = is_debug

        if self.is_debug:
            self.epoch = 1

    def model_params(self) -> Dict:
        return {}


class GoogleBrainLSTMTrainDataset(Dataset):
    def __init__(self, X: DataFrame, y: ndarray, categorical_features: List[str], continuous_features: List[str], target_dict: Dict) -> None:
        self.X = X
        self.y = y
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.target_dict = target_dict

        self.groups = X.groupby("breath_id").groups
        self.keys = list(self.groups.keys())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        idx = self.groups[self.keys[index]]
        X = self.X.iloc[idx]
        y = self.y.iloc[idx]["pressure"]

        categorical_X = torch.LongTensor(X[self.categorical_features].values)
        continuous_X = torch.FloatTensor(X[self.continuous_features].values)
        u_out = torch.LongTensor(X["u_out"].values)
        label = [self.target_dict[i] for i in y.values]
        label = torch.LongTensor(label)

        return categorical_X, continuous_X, u_out, label


class GoogleBrainLSTMTestDataset(Dataset):
    def __init__(self, X: DataFrame, categorical_features: List[str], continuous_features: List[str]) -> None:
        self.X = X
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        self.groups = X.groupby("breath_id").groups
        self.keys = list(self.groups.keys())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        idx = self.groups[self.keys[index]]
        X = self.X.iloc[idx]

        categorical_X = torch.LongTensor(X[self.categorical_features].values)
        continuous_X = torch.FloatTensor(X[self.continuous_features].values)

        return categorical_X, continuous_X


class LSTMTorchModel(nn.Module):
    def __init__(self, config: GoogleBrainLSTMConfig):
        super().__init__()

        self.config = config

        input_dim = 4 + len(self.config.continuous_features)
        dense_dim = 64
        lstm_dim = 600
        logits_dim = 256

        # self.r_emb = nn.Embedding(3, 2, padding_idx=0)
        # self.c_emb = nn.Embedding(3, 2, padding_idx=0)
        self.rc_emb = nn.Embedding(9, 4, padding_idx=0)
        # self.rc_sum_emb = nn.Embedding(8, 4, padding_idx=0)
        # self.rc_mul_emb = nn.Embedding(8, 4, padding_idx=0)

        self.mlp = nn.Sequential(nn.Linear(input_dim, dense_dim), nn.LayerNorm(dense_dim), nn.ReLU())

        self.lstm1 = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_dim * 2, 500, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(1000, 400, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(800, logits_dim, batch_first=True, bidirectional=True)

        self.ln = nn.LayerNorm(logits_dim * 2)

        self.logits = nn.Sequential(
            nn.Linear(logits_dim * 2, logits_dim * 2),
            nn.LayerNorm(logits_dim * 2),
            nn.ReLU(),
            nn.Linear(logits_dim * 2, 950),
        )

        ##### Init Params

        init_range = 0.1
        # self.r_emb.weight.data.uniform_(-init_range, init_range)
        # self.c_emb.weight.data.uniform_(-init_range, init_range)
        self.rc_emb.weight.data.uniform_(-init_range, init_range)
        # self.rc_sum_emb.weight.data.uniform_(-init_range, init_range)
        # self.rc_mul_emb.weight.data.uniform_(-init_range, init_range)

        for _, m in self.named_modules():
            if isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param.data)
                    else:
                        nn.init.normal_(param.data)
            elif isinstance(m, (nn.Conv1d, nn.Linear)):
                for name, param in m.named_parameters():
                    if name == "weight":
                        nn.init.kaiming_normal_(param.data)

    def forward(self, categorical_X, continuous_X):
        bs = continuous_X.size(0)

        # r_emb = self.r_emb(categorical_X[:, :, 0]).view(bs, 80, -1)
        # c_emb = self.c_emb(categorical_X[:, :, 1]).view(bs, 80, -1)
        rc_emb = self.rc_emb(categorical_X[:, :, 0]).view(bs, 80, -1)
        # rc_sum_emb = self.rc_sum_emb(categorical_X[:, :, 2]).view(bs, 80, -1)
        # rc_mul_emb = self.rc_mul_emb(categorical_X[:, :, 3]).view(bs, 80, -1)

        x = torch.cat((rc_emb, continuous_X), 2)
        # x = continuous_X

        features = self.mlp(x)

        features, _ = self.lstm1(features)
        features, _ = self.lstm2(features)
        features, _ = self.lstm3(features)
        features, _ = self.lstm4(features)

        pred = self.logits(features)

        return pred


class GoogleBrainLSTM(ModelWrapper):
    def __init__(self, config: GoogleBrainLSTMConfig, metrics: Metrics, score: Metrics, file_logger: Logger, std_logger: Logger) -> None:
        super().__init__(config, metrics, score, file_logger, std_logger)
        if config.is_debug:
            self.std_logger.info("This is Debug Mode.")

    def build(self):
        self._seed_everything(self.config.seed)

        self.model = LSTMTorchModel(self.config)
        self.model.to(self.config.device)
        # self.optimizer = Adam(self.model.parameters(), lr=5e-3)
        self.optimizer = AdamW(self.model.parameters(), lr=1.5e-3, weight_decay=5e-2)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp)

    def load(self):
        model = LSTMTorchModel(self.config)
        state = torch.load(self.config.save_model_path)
        model.load_state_dict(state)
        model.to(self.config.device)
        self.model = model

    def fit(
        self,
        X_train: Union[DataFrame, ndarray],
        y_train: Union[Series, ndarray],
        X_valid: Union[DataFrame, ndarray],
        y_valid: Union[Series, ndarray],
        opt_params: Optional[Dict] = None,
        **kwargs,
    ) -> FitResult:
        self.file_logger.info("LSTM training start.")

        mask = X_valid[X_valid["u_out"] == 0].index

        train_dataset = GoogleBrainLSTMTrainDataset(
            X_train, y_train, self.config.categorical_features, self.config.continuous_features, self.config.target_dict
        )
        valid_dataset = GoogleBrainLSTMTrainDataset(
            X_valid, y_valid, self.config.categorical_features, self.config.continuous_features, self.config.target_dict
        )

        train_dataloader = DataLoader(
            train_dataset, self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, pin_memory=False, drop_last=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, pin_memory=False, drop_last=False
        )

        # Scheduler

        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.T_max, eta_min=self.config.min_lr, last_epoch=-1)

        # self.scheduler = ExponentialLR(self.optimizer, gamma=self.config.gamma)

        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=40, T_mult=1, eta_min=5e-6, last_epoch=-1)

        # self.scheduler = OneCycleLR(
        #     optimizer=self.optimizer,
        #     pct_start=0.2,
        #     div_factor=1e3,
        #     max_lr=5e-3,
        #     epochs=self.config.epoch,
        #     steps_per_epoch=len(train_dataloader),
        # )

        # num_train_steps = int(len(train_dataloader) * self.config.epoch)
        # num_warmup_steps = int(num_train_steps / 10)
        # self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)

        best_score = np.inf
        best_valid_loss = np.inf
        best_oof_prediction = None

        for epoch in range(self.config.epoch):
            timer = Timer()
            timer.start()
            train_loss = self._train(epoch + 1, train_dataloader)
            oof_prediction, valid_loss = self._valid(epoch + 1, valid_dataloader)
            score = self.score.execute(y_valid.values[mask], oof_prediction[mask])
            timer.end()

            self.file_logger.info(f"Epoch [{epoch+1}/{self.config.epoch}]: Score -> {score} ({timer.result}s)")
            self.std_logger.info(f"Epoch [{epoch+1}/{self.config.epoch}]: Score -> {score} ({timer.result}s)")

            self.scheduler.step()

            if score < best_score:
                self.save_model()
                best_score = score
                best_valid_loss = valid_loss
                best_oof_prediction = oof_prediction

        torch.cuda.empty_cache()
        gc.collect()

        return FitResult(self.model, best_oof_prediction, best_valid_loss, best_score, None)

    def predict(self, X_test: Union[DataFrame, ndarray], **kwargs) -> ndarray:
        test_dataset = GoogleBrainLSTMTestDataset(X_test, self.config.categorical_features, self.config.continuous_features)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        test_predictions = []
        tk0 = tqdm(enumerate(test_loader), total=len(test_loader))

        self.model.eval()
        for step, (categorical_X, continious_X) in tk0:
            categorical_X, continious_X = categorical_X.to(self.config.device), continious_X.to(self.config.device)
            with torch.no_grad():
                test_prediction = self.model(categorical_X, continious_X)

            test_prediction = test_prediction.reshape(-1, 950).softmax(1)
            test_prediction = torch.sum(torch.tensor(self.config.unique_targets).to(self.config.device) * test_prediction, axis=1)
            test_predictions.append(test_prediction.cpu().numpy())

        test_predictions = np.concatenate(test_predictions)

        return test_predictions

    def optimize(
        self,
        X_train: Union[DataFrame, ndarray],
        y_train: Union[Series, ndarray],
        X_valid: Union[DataFrame, ndarray],
        y_valid: Union[Series, ndarray],
        direction: str,
        n_trials: int,
        **kwargs,
    ) -> Dict:
        pass

    def save_model(self):
        self.file_logger.info("save model ...")
        torch.save(self.model.state_dict(), self.config.save_model_path)

    def _train(self, epoch: int, train_loader: DataLoader) -> float:
        losses = AverageMeter()

        self.model.train()
        self.optimizer.zero_grad()
        for step, (categorical_X, continuous_X, u_out, label) in enumerate(train_loader):
            categorical_X = categorical_X.to(self.config.device)
            continuous_X = continuous_X.to(self.config.device)
            label = label.to(self.config.device)

            with torch.cuda.amp.autocast(enabled=self.config.use_amp):
                prediction = self.model(categorical_X, continuous_X)
                _prediction = prediction.reshape(-1, 950).softmax(1)
                _prediction = torch.sum(torch.tensor(self.config.unique_targets).to(self.config.device) * _prediction, axis=1).reshape(-1, 80)

                mask = u_out == 0
                targets = torch.tensor(self.config.unique_targets[label[mask].tolist()]).to(self.config.device)

                loss_cls = self.label_smoothing_loss(prediction[mask], label[mask])
                loss_reg = 0.5 * nn.SmoothL1Loss(beta=0.1)(_prediction[mask], targets)
                loss = loss_cls + loss_reg

            # loss = self.criterion(prediction, y)
            losses.update(loss.item(), self.config.batch_size)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            # self.scheduler.step()

            if step % self.config.print_freq == 0 or step == (len(train_loader) - 1):
                self.file_logger.info(
                    f"[Train] Epoch: [{epoch}][{step + 1} / {len(train_loader)}] \n"
                    f"Loss: [val: {losses.val}] [avg: {losses.avg}]  [lr: {self.scheduler.get_last_lr()}] \n"
                )

        return losses.avg

    def _valid(self, epoch: int, valid_loader: DataLoader):
        losses = AverageMeter()
        predictions = []

        self.model.eval()
        for step, (categorical_X, continuous_X, u_out, label) in enumerate(valid_loader):
            categorical_X = categorical_X.to(self.config.device)
            continuous_X = continuous_X.to(self.config.device)
            label = label.to(self.config.device)

            with torch.no_grad():
                prediction = self.model(categorical_X, continuous_X)

            _prediction = prediction.reshape(-1, 950).softmax(1)
            _prediction = torch.sum(torch.tensor(self.config.unique_targets).to(self.config.device) * _prediction, axis=1)

            mask = u_out == 0
            targets = torch.tensor(self.config.unique_targets[label[mask].tolist()]).to(self.config.device)

            loss_cls = self.label_smoothing_loss(prediction[mask], label[mask])
            loss_reg = 0.5 * nn.SmoothL1Loss(beta=0.1)(_prediction.reshape(-1, 80)[mask], targets)
            loss = loss_cls + loss_reg

            losses.update(loss.item(), self.config.batch_size)

            predictions.append(_prediction.cpu().numpy())

            if step % self.config.print_freq == 0 or step == (len(valid_loader) - 1):
                self.file_logger.info(f"[Valid] Epoch: [{epoch}][{step + 1} / {len(valid_loader)}] Loss: [val: {losses.val}] [avg: {losses.avg}] \n")

        predictions = np.concatenate(predictions)

        return predictions, losses.avg

    def label_smoothing_loss(self, y_pred, y_true):
        criterion = nn.CrossEntropyLoss()

        loss = criterion(y_pred.reshape(-1, 950), y_true.reshape(-1))

        for lag, w in [(1, 0.4), (2, 0.2), (3, 0.1), (4, 0.1)]:
            # negative lag loss
            # if target < 0, target = 0
            neg_lag_target = F.relu(y_true.reshape(-1) - lag)
            neg_lag_target = neg_lag_target.long()
            neg_lag_loss = criterion(y_pred.reshape(-1, 950), neg_lag_target)

            # positive lag loss
            # if target > 949, target = 949
            pos_lag_target = 949 - F.relu((949 - (y_true.reshape(-1) + lag)))
            pos_lag_target = pos_lag_target.long()
            pos_lag_loss = criterion(y_pred.reshape(-1, 950), pos_lag_target)

            loss += (neg_lag_loss + pos_lag_loss) * w

        return loss

    def _seed_everything(self, seed=1013):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
