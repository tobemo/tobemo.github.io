import argparse
import importlib
import os
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import yaml
from lightning.pytorch import LightningDataModule, LightningModule


OUTPUT_BASE = Path(
    os.getenv("OUTPUT_BASE", "output")
)
RUNS = Path(
    os.getenv("RUNS", "runs")
)


def load_config(fp: Path) -> dict:
    with open(fp, "r") as f:
        return yaml.safe_load(f)


def load_config_from_checkpoint(ckpt: Path) -> dict:
    # assumes ../vesion_x/checkpoints/ckpt structure 
    # with config at the same level of checkpoints
    return load_config(
        ckpt.parent.parent / "config.yaml"
    )


def get_module_used(class_path: str) -> LightningModule | LightningDataModule:
    return getattr(
        importlib.import_module(class_path.rsplit('.', 1)[0]),
        class_path.rsplit('.', 1)[-1]
    )


def get_checkpoint(name: str, version: int, ckpt: str = 'last') -> Path:
    return RUNS / f"{name}/version_{version}/checkpoints/{ckpt}.ckpt"


def prep_model_for_export(name: str, version: int, ckpt: str = 'last') -> tuple[LightningModule, torch.Tensor]:
    """Prepare a lightning module for export to onnx format.
    
    The mean and standard deviation used during pre processing are written to data.yaml file
    in the folder 'output' by default. Set the environment variable "OUTPUT_BASE" if another folder is desired.

    Args:
        name (str): What model to load. Must exist in './runs/'.
        The folder runs must following the lightning_runs folder structure.
        Set the environment variable "RUNS" if another folder is used.
        version (int): What version to load.
        ckpt (str, optional): What checkpoint to load. Defaults to 'last'.

    Returns:
        tuple[LightningModule, torch.Tensor]: The model to convert and a sample input tensor of onnx.
    """
    ckpt = get_checkpoint(name=name, version=version, ckpt=ckpt)
    fp_base = OUTPUT_BASE / name
    fp_base.mkdir(exist_ok=True, parents=True)
    
    # get model and datamodule used from config that lightning-cli creates
    used_configuration = load_config_from_checkpoint(ckpt)
    data_module = get_module_used(class_path=used_configuration['data']['class_path'])
    lightning_module = get_module_used(class_path=used_configuration['model']['class_path'])
    
    # load and prep model and datamodule
    model = lightning_module.load_from_checkpoint(ckpt)
    model.cpu()
    model.eval()
    data = data_module.load_from_checkpoint(ckpt)

    # store constants needed for inference preprocessing

    preprocessing_constants = {
        "mean": float(data.mean),
        "std": float(data.std),
    }
    with open(fp_base / 'data.yaml', 'w') as outfile:
        yaml.dump(preprocessing_constants, outfile)
    
    # onnx needs a reference input;
    # future inputs passed to the resulting onnx model
    # that differ from this shape will cause an error
    sample = torch.ones(1, model.hparams.input_channels, data.length) # TODO: set as .example_input_array in LightningModule
    return model, sample


def to_onnx(name: str, model: LightningModule, sample: torch.Tensor) -> None:
    fp_base = OUTPUT_BASE / name
    fp_model = fp_base / "model.onnx"
    fp_model_opt = fp_base / "model_opt.onnx"
    
    # use pytorch lightning to create a default onnx model
    model.to_onnx(fp_model, sample, export_params=True)
    
    # use onnxruntime to convert this to an optimized version
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = str(fp_model_opt)
    
    # convert
    session = onnxruntime.InferenceSession(fp_model, sess_options)
    input_name = session.get_inputs()[0].name
    ort_inputs = {input_name: np.ones(session.get_inputs()[0].shape, dtype=np.float32)}
    session.run(None, ort_inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert a LightningModule to onnx format. \
            This assumes models are stored in the folder called runs and use the default\
                lightning logging folder structure.'
    )
    parser.add_argument('-n','--name', help='Name of lightning run to use.', required=True)
    parser.add_argument('-v','--version', help='Version to convert.', required=True)
    parser.add_argument('-c','--ckpt', default='last.ckpt', help='In case a checkpoint other than last is desired.', required=False)
    args = vars(parser.parse_args())
    
    to_onnx(
        name=args['name']
        *prep_model_for_export(
            **args
        )
    )
