# Didactic meme
A modelling suite with extra focus on pytorch

* Speed up the modelling process
* Increase traceability of trained models
* Easier model comparison and highscores
* Visualize model predictions
* Easily expose model via web api

Planned features:
- [x] Model config
- [ ] Standard training loop
- [x] Setup loggers
- [x] Command line config and training
- [ ] Visualize blackbox solution
- [ ] Web api helper
- [x] Training helper functions

## Usage

### Standard training loop

    def get_loss(batch):
        # ...
        return loss

    model_suite.fit(get_loss, model, optimizer, train_loader,
        validate_loader, config)


## TODO

- [x] Save models by epoch
- [ ] Score models and list highscore
- [x] Tensorboardx
- [ ] Need to handle custom pre-processing
