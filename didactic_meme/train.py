import os
import time
import logging
import torch
import tensorboardX as tbx
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, train_loader, validate_loader, model, optimizer, config):
        self.logger = logging.getLogger(__name__)

        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.model = model
        self.optimizer = optimizer
        self.config = config

        self.device = torch.device('cuda' if self.config.cuda else 'cpu')
        self.tb = tbx.SummaryWriter(log_dir=config.model_dir)

    def save_checkpoint(self, epoch):
        save_dir = self.config.get_checkpoint_dir()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(dict(
            epoch=epoch,
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        ), self.config.get_checkpoint_path(epoch))

    def add_tensorboard_scalars(self, scalars_dict, epoch):
        for name, scalars in scalars_dict.items():
            self.tb.add_scalars(name, scalars, epoch)

    def get_logger_text(self, epoch, time_spent, epoch_summary):
        n_epochs_chars = len(str(self.config.n_epochs))
        return ' '.join([
            f'[Epoch {epoch:>{n_epochs_chars}}/{self.config.n_epochs}]',
            f'({time_spent:.1f}s)',
            ', '.join([
                f'{name}: ({self.get_logger_values_text(values)})'
                for name, values in epoch_summary.items()
            ]),
        ])

    def get_logger_values_text(self, values):
        return ', '.join([f'{type}: {value:.6f}' for type, value in values.items()])

    def zip_dicts(self, *dicts):
        stacked_results = {}
        for d in dicts:
            for key, value in d.items():
                if key not in stacked_results:
                    stacked_results[key] = []
                stacked_results[key].append(value)
        return stacked_results

    def merge_add_level(self, **stacked_results):
        results = {}
        for type, values in stacked_results.items():
            for key, value in values.items():
                if key not in results:
                    results[key] = {}
                results[key][type] = value
        return results

    def summarize_epoch(self, train_results, validate_results, epoch):
        train_results, validate_results = self.zip_dicts(*train_results), self.zip_dicts(*validate_results)

        train_results = {key: sum(value)/len(self.train_loader.dataset) for key, value in train_results.items()}
        validate_results = {key: sum(value)/len(self.validate_loader.dataset) for key, value in validate_results.items()}

        epoch_results = self.merge_add_level(train=train_results, validate=validate_results)
        self.add_tensorboard_scalars(epoch_results, epoch)

        return epoch_results

    def train_batch(self, features, labels):
        log_prob = self.model(features).log_prob(labels)
        loss = -log_prob.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return dict(
            log_prob=log_prob.sum(),
        )

    def evaluate_batch(self, features, labels):
        return dict(
            log_prob=self.model(features).log_prob(labels).sum(),
        )


    def train(self):
        if self.config.seed is None:
            self.logger.info(f'no seed')
        else:
            torch.manual_seed(self.config.seed)
            self.logger.info(f'seed: {self.config.seed}')

        for epoch in range(1, self.config.n_epochs + 1):
            start_time = time.time()
            self.model.train()
            train_results = [self.train_batch(*[d.to(self.device) for d in batch]) for batch in self.train_loader]

            self.model.eval()
            with torch.no_grad():
                validate_results = [self.evaluate_batch(*[d.to(self.device) for d in batch]) for batch in self.validate_loader]

            if epoch == self.config.n_epochs or (
                self.config.save_interval is not None and
                epoch % self.config.save_interval == 0
            ):
                self.save_checkpoint(epoch)
                self.logger.info('saved checkpoint')

            epoch_results = self.summarize_epoch(train_results, validate_results, epoch)

            time_spent = time.time() - start_time
            self.logger.info(self.get_logger_text(epoch, time_spent, epoch_results))


class MultipleOptimizers:
    def __init__(self, **optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        return {key: optimizer.state_dict() for key, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dicts):
        for key, state_dict in state_dicts.items():
            self.optimizers[key].load_state_dict(state_dict)

    def __getattr__(self, name):
        if name in self.optimizers:
            return self.optimizers[name]
        return getattr(self.optimizers, name)
