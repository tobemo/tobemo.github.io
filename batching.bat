@ECHO OFF
CALL .venv\Scripts\activate

set model=reference
python train.py fit --trainer ./configs/trainer_%model%.yaml --model ./configs/model_%model%.yaml --data ./configs/data_%model%.yaml --trainer.logger.init_args.name %model%

python train_and_shutdown.py fit --trainer ./configs/trainer_%model%.yaml --model ./configs/model_%model%.yaml --data ./configs/data_%model%.yaml --trainer.logger.init_args.name %model%
