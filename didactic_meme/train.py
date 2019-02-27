import os
import time
import logging
import torch
import tensorboardX as tbx
logger = logging.getLogger(__name__)


def save_checkpoint(save_dict, config):

    save_dir = os.path.join(config.model_dir, 'save') # TODO: move to model-suite?
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(save_dict, os.path.join(save_dir, get_save_filename(epoch, config.n_epochs)))

def add_tb_scalars(tb, scalars_dict, epoch):
    for name, scalars in scalars_dict.items():
        tb.add_scalars(name, scalars, epoch)

def get_save_filename(epoch, n_epochs):
    n_epochs_chars = len(str(n_epochs))
    return f'epoch{epoch:0{n_epochs_chars}d}.pth'

def get_logger_text(epoch, n_epochs, time_spent, log_dict):
    n_epochs_chars = len(str(n_epochs))
    return ' '.join([
        f'[Epoch {epoch:>{n_epochs_chars}}/{n_epochs}]',
        f'({time_spent:.1f}s)',
        ', '.join([
            f'{name}: ({get_logger_values_text(values)})'
            for name, values in log_dict.items()
        ]),
    ])

def get_logger_values_text(values):
    return ', '.join([f'{type}: {value:.6f}' for type, value in values.items()])

def merge_epoch_results(**results_dict):
    merged_results = {}
    for type, results in results_dict.items():
        for key, value in results.items():
            if key not in merged_results:
                merged_results[key] = {}
            merged_results[key][type] = value
    return merged_results

def epoch(batch_func, data_loader, config, sample_weight=None):
    '''
    Sums results from batch_func and finally divides by length of
    dataset or sum of sample weights
    '''
    device = torch.device('cuda' if config.cuda else 'cpu')

    results = None
    for batch in data_loader:
        batch_results = batch_func(*[d.to(device) for d in batch])

        if results is None:
            results = batch_results
        else:
            for key, value in results.items():
                results[key] += batch_results[key]

    if sample_weight is None:
        for key, value in results.items():
            results[key] /= len(data_loader.dataset) # TODO: does not handle drop_last=True?
    else:
        for key, value in results.items():
            if key != sample_weight:
                results[key] /= results[sample_weight]
        del results[sample_weight]

    return results

def train(train_func, eval_func, train_loader, validate_loader, model, optimizer, config, sample_weight=None):
    torch.manual_seed(config.seed)
    logger.info(f'seed: {config.seed}')

    tb = tbx.SummaryWriter(log_dir=config.model_dir)

    for epoch_number in range(1, config.n_epochs + 1):
        start_time = time.time()
        model.train()
        train_results = epoch(train_func, train_loader, config, sample_weight=sample_weight)

        model.eval()
        with torch.no_grad():
            validate_results = epoch(eval_func, validate_loader, config, sample_weight=sample_weight)

        epoch_results = merge_epoch_results(train=train_results, validate=validate_results)
        add_tb_scalars(tb, epoch_results, epoch_number)

        if config.save_interval is not None and epoch_number % config.save_interval == 0:
            save_checkpoint(dict(
                epoch=epoch_number,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
            ), config)

        time_spent = time.time() - start_time
        logger.info(get_logger_text(epoch_number, config.n_epochs, time_spent,
            epoch_results))

class MultipleOptimizers:
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        for optimizer in self.optimizers:
            optimizer.state_dict()

    def load_state_dict(self, state_dicts):
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)
