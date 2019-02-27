import os
import logging
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


def train_epoch(loss_func, data_loader, epoch):
    model.train()

    for batch_idx, batch in enumerate(data_loader):
        batch = [b.to(device) for b in batch]
        loss = loss_func(batch, epoch)
