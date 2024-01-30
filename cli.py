from copy import deepcopy
from pathlib import Path
from typing import Dict, Set

import optuna
import yaml
from jsonargparse import Namespace
from lightning.pytorch import Trainer
from lightning.pytorch.cli import (LightningArgumentParser, LightningCLI,
                                   instantiate_class)
from lightning.pytorch.trainer import Trainer
from optuna.integration.pytorch_lightning import \
    PyTorchLightningPruningCallback

from data.datamodule import BoringDataModule
from model.base import BoringModel


class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)
    
    def optimize(self, *args, **kwargs) -> None:
        """Alias for fit."""
        self.fit(*args, **kwargs)


def _trial_suggest_from_config(trial: optuna.Trial, optuna: dict, config: Namespace, name: str) -> Namespace:
    if 'class_path' in optuna and isinstance(optuna['class_path'], list):
        trial.suggest_categorical(
            name=name + '.' + 'class_path',
            choices=optuna['class_path'],
        )
    if 'init_args' in optuna:
        for arg, values in optuna['init_args'].items():
            fn = getattr(trial, f"suggest_{values['type']}")
            val = fn(name=name + '.' + arg, **values['kwargs'])
            setattr(config, 'init_args.' + arg, val)
    return config


class EnhancedCli(LightningCLI):
    study: optuna.Study = None
    to_monitor: str
    _optuna_description: str = "Runs a hyperparameter search."
    _last_trainer: MyTrainer = None
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            trainer_class=MyTrainer,
            run=True, # has to be True
            parser_kwargs={'optimize': {'description': self._optuna_description}}
        )

    #### Methods shown below are called in order ####
    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        # add 'optimize' to the list of available subcommands
        return {
            "fit": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "optimize": {"model", "train_dataloaders", "val_dataloaders", "datamodule"},
            "validate": {"model", "dataloaders", "datamodule"},
            "test": {"model", "dataloaders", "datamodule"},
            "predict": {"model", "dataloaders", "datamodule"},
        }
    
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        # add optuna args but only to parser that deals with subcommand 'optimize'
        # currently there is no better way to check which subcommand-parser is
        # given other than manually set and check the parser description
        if parser.description == self._optuna_description:
            parser.add_argument_group('optuna')
            # metric to monitor
            parser.add_argument("--optuna.monitor")
            # new study settings; optuna's defaults are used
            parser.add_argument("--optuna.study.storage", default="sqlite:///optuna.db")
            parser.add_argument("--optuna.study.sampler", default=None)
            parser.add_argument("--optuna.study.pruner", default=None)
            parser.add_argument("--optuna.study.study_name", default=None)
            parser.add_argument("--optuna.study.direction", default='minimize')
            parser.add_argument("--optuna.study.load_if_exists", default=True)
            parser.add_argument("--optuna.study.directions", default=None)
            # optimize call of a study settings; again using optuna's defaults
            parser.add_argument("--optuna.optimize.n_trials", default=10)
            parser.add_argument("--optuna.optimize.timeout", default=None)
            parser.add_argument("--optuna.optimize.n_jobs", default=1)
            parser.add_argument("--optuna.optimize.gc_after_trial", default=False)
            parser.add_argument("--optuna.optimize.show_progress_bar", default=False)
            
            # ensure there is at least an empty for both model and data
            # or else jsonargparse complains
            parser.add_argument("--optuna.model", default={})
            parser.add_argument("--optuna.data", default={})
    
    def before_instantiate_classes(self) -> None:
        # create/load study if optimizing
        if self.subcommand == 'optimize':
            # store metric to monitor
            self.to_monitor = self._get(self.config, "optuna.monitor")
            if self.to_monitor is None:
                raise ValueError('Metric for optuna to monitor is not set.')
            
            # get settings and possibly instantiate non-default sampler and pruner
            study_settings = self._get(self.config, "optuna.study")
            study_settings.sampler = study_settings.sampler if study_settings.sampler is None else \
                instantiate_class(args=(), init=study_settings.sampler)
            study_settings.pruner = study_settings.pruner if study_settings.pruner is None else \
                instantiate_class(args=(), init=study_settings.pruner)
            
            # create new or load existing optuna-study
            self.study = optuna.create_study(**study_settings)
        
    def before_optimize(self) -> None:
        # inject one or more self.objective() calls before Trainer.optimize() is called
        # then turn Trainer.optimize() into a no-op
        optimize_settings = self._get(self.config, "optuna.optimize")
        self.study.optimize(
            self.objective,
            **optimize_settings
        )

        # prevent training run
        self.trainer.optimize = lambda *args, **kwargs: None
    
    def objective(self, trial: optuna.trial.Trial) -> float:
        # one training run using parameters suggested to test for this trial
        # if configs are not present in optuna original configs are used
        
        # store parameters and best trial of last run
        if self._last_trainer is not None:
            self._store_intermediate_results(self._last_trainer)
        
        # model suggestion
        model_config = _trial_suggest_from_config(
            trial=trial,
            optuna=self._get(self.config, "optuna.model"),
            config=self._get(self.config, "model"),
            name='model',
        )
        self.model = instantiate_class(
            args=(),
            init=model_config,
        )
        
        # datamodule suggestion
        # don't initialize datamodule more than once if not needed
        data_config = self._get(self.config, "optuna.data")
        if len(data_config) > 0:
            data_config = _trial_suggest_from_config(
                trial=trial,
                optuna=data_config,
                config=self._get(self.config, "data"),
                name='data',
            )
            self.datamodule = instantiate_class(
                args=(),
                init=data_config,
            )
        
        # don't remember why but a deepcopy was important when doing more than 1 run
        trainer = deepcopy(self.trainer)
        trainer.callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor=self.to_monitor)
        )
        self._last_trainer = trainer # ! store before training or risk not being set when trial is pruned
        
        # train as usual; self.model and self.datamodule where adjusted above
        # and will be returned by '_prepare_subcommand_kwargs'
        fn_kwargs = self._prepare_subcommand_kwargs('optimize')
        trainer.optimize(**fn_kwargs)
        
        # return last 'to_monitor' value to optuna optimize
        return trainer.callback_metrics[self.to_monitor].item()
    
    def _store_intermediate_results(self, trainer: MyTrainer = None) -> None:
        """"Write what are currently the best parameters to file.
        
        Args:
            trainer (MyTrainer, optional): Results are written to log dir of this trainer.
            Defaults to None in which case self.trainer is used.
        """
        trainer = trainer or self.trainer
        trials = self.study.best_trials
        results = []
        for trial in trials:
            results.append({
                'best_trial': trial.number,
                'relevant_version': trainer.logger.version,
                'value': trial.values,
                'params': trial.params,
            })
        n_finished_trials = len(
            [t for t in self.study.get_trials(deepcopy=False) if t.state.is_finished()]
        )
        with open(Path(trainer.logger.log_dir) / f"trial_{n_finished_trials-1}.yaml", 'w') as f:
            yaml.safe_dump(results, f)
    
    def after_optimize(self) -> None:
        # called right after Trainer.optimize() is called
        # and by extension called after optimization has finished
        
        # print results
        print("Number of finished trials: {}".format(len(self.study.trials)))
        print("Best trial(s):")
        trials = self.study.best_trials
        for trial in trials:
            print(" ", f"Trial {trial.number}")
            print("  ", f"Value(s): {','.join([f'{v:.5f}' for v in trial.values])}")
            print("  ", "Params: ")
            for key, value in trial.params.items():
                print("   ", f"{key}: {value}")
        
        # store results
        self._store_intermediate_results(self._last_trainer)


def get_cli() -> None:
    cli = EnhancedCli(
        model_class=BoringModel, subclass_mode_model=True,
        datamodule_class=BoringDataModule, subclass_mode_data=True,
    )


if __name__ == "__main__":
    get_cli()

