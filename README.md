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
