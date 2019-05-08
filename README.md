# Didactic meme
A modelling suite with extra focus on pytorch

* Speed up the modelling process
* Increase traceability of trained models
* Easier model comparison and highscores
* Visualize model predictions
* Easily expose model via web api
* Hash train, validate, and test datasets separately


Planned features:
- [x] Model config
- [x] Standard training loop
- [x] Setup loggers
- [x] Command line config and training
- [ ] Visualize blackbox solution
- [ ] Web api helper
- [x] Training helper functions
- [ ] Intra-epoch logging

## TODO

- [x] Reconsider how api and helpers work
- [x] Save models by epoch
- [ ] Score models and list highscore
- [x] Tensorboardx
- [ ] Need to handle custom pre-processing
- [ ] Continue training/tuning from checkpoint (hash initial model)
- [ ] Tabular logging like skorch

## Usage
Draft of how the library should be used.

### Overly simplistic training
No custom code or metrics

    def train(train_ds, validate_ds, config):
        # create train_loader, validate_loader, model, and optimizer
        model_suite.Trainer(train_loader, validate_loader, model, optimizer, config).train()

### Standard training loop
Custom metric

    class Trainer(model_suite.Trainer):
        def train_batch(self, features, labels):
            log_prob = model(features).log_prob(labels)
            loss = -log_prob.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return dict(
                log_prob=log_prob.sum(),
                accuracy=get_accuracy(features, labels, self.model).mean(),
            )

        def evaluate_batch(self, features, labels):
            return dict(
                log_prob=model(features).log_prob(labels).sum(),
                accuracy=get_accuracy(features, labels, self.model).mean(),
            )

        def summarize_epoch(train_results, validate_results, epoch):
            train_results, validate_results = self.zip_dicts(train_results), self.zip_dicts(validate_results)

            train_results = {key: sum(value)/len(self.train_loader.dataset) for key, value in train_results.items()}
            validate_results = {key: sum(value)/len(self.validate_loader.dataset) for key, value in validate_results.items()}

            epoch_results = self.merge_add_level(train=train_results, validate=validate_results)
            self.add_tb_scalars(epoch_results, epoch)

            return {key: epoch_results[key] for key in ['log_prob']}

    def train(train_ds, validate_ds, config):
        # create train_loader, validate_loader, model, and optimizer
        Trainer(train_loader, validate_loader, model, optimizer, config).train()


### Generative Adverserial Model

    class Trainer(model_suite.Trainer):
        def train_batch(self, features, boards):
            discriminator_log_prob = self.model.get_discriminator_log_prob(features, boards)
            generator_log_prob = self.model.get_generator_log_prob(features)

            discriminator_loss = -discriminator_log_prob.mean()
            self.optimizer.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.optimizer.discriminator_optimizer.step()

            generator_loss = -generator_log_prob.mean()
            self.optimizer.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.optimizer.generator_optimizer.step()

            return dict(
                discriminator_log_prob=discriminator_log_prob.sum(),
                generator_log_prob=generator_log_prob.sum(),
            )

        def evaluate_batch(self, features, boards):
            return dict(
                discriminator_log_prob=self.model.get_discriminator_log_prob(features, boards).sum(),
                generator_log_prob=self.model.get_generator_log_prob(features).sum(),
            )


    def train(train_ds, validate_ds, config):
        # setup data loaders, model and optimizers

        optimizer = model_suite.MultipleOptimizers(
            generator_optimizer=...,
            discriminator_optimizer=...,
        )

        Trainer(train_loader, validate_loader, model, optimizer, config).train()
