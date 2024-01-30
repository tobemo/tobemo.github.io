# Ready to go pytorch lightning project
## Usage
1) `git clone https://github.com/tobemo/ReferenceProject.git`
2) launch setup.bat:
    - creates a virtual environment
    - installs everything in requirements.txt
3) start working

## Noteworthy
- Implements Lightning CLI.
- Intergrates optuna into Lightning CLI.
- Correctly logs hyperparameters for easy interpretation with tensorboard.
- Defaults to the [AdaBelief](https://juntang-zhuang.github.io/adabelief/) optimizer.
- Implements a [cosine annealing with warm restarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html) learning rate scheduler of which the amplitude decreases every epoch.

## Examples
**Normal training:**
> `python .\train.py fit --trainer .\configs\trainer_reference.yaml --model .\configs\model_reference.yaml --data .\configs\data_reference.yaml`

**Optimizing:**
> `python .\train.py optimize --trainer .\configs\trainer_reference.yaml --model .\configs\model_reference.yaml --data .\configs\data_reference.yaml --config .\configs\optimizer_reference.yaml`

## On optuna integration
> Check out configs/optimizer_reference.yaml

- All `suggest_*` of the [optuna trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial) object are supported, together with their respective arguments.
- Both `init_args` and `class_path` can be suggested by optuna, just make sure to use the type `categorical` when setting `class_path`. Defaults to settings in model config so not everything needs to be set.
An example:
```YAML
optuna:
    ...
    data:
      class_path:
          type: categorical
          choices: [data.datamodule.BoringDataModule, data.datamodule.OtherDataModule]
      init_args:
        batch_size
            type: int
            kwargs:
              low: 32
              high: 256
    model:
      init_args:
          lr:
            type: float
            kwargs:
              low: 1e-3
              high: 1e-3
              log: True
```
