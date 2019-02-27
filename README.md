# Didactic meme
A modelling suite with extra focus on pytorch

* Speed up the modelling process
* Increase traceability of trained models
* Easier model comparison and highscores
* Visualize model predictions
* Easily expose model via web api

Planned features:
- [x] Model config
- [x] Standard training loop
- [x] Setup loggers
- [x] Command line config and training
- [ ] Visualize blackbox solution
- [ ] Web api helper
- [x] Training helper functions

## Usage

### Standard training loop

    def train(train_ds, validate_ds, config):

        # define model, optimizer, data_loaders

        def train_batch(features, labels):
            log_prob = get_log_prob(features, labels, model)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            return dict(
                log_prob=discriminator_log_prob.sum(),
            )

        def eval_batch(features, boards):
            return dict(
                discriminator_log_prob=get_log_prob(features, labels, model).sum(),
            )

        model_suite.train(train_batch, eval_batch, train_loader, validate_loader, model, optimizer, config)


## TODO

- [ ] Reconsider how api and helpers work
- [x] Save models by epoch
- [ ] Score models and list highscore
- [x] Tensorboardx
- [ ] Need to handle custom pre-processing
