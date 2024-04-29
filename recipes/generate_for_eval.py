# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import h5py
import itertools
import sys
import time
from typing import Any, Dict

import torch
from omegaconf import DictConfig

import pandas as pd

from torch import nn

from torchtune import config, utils

logger = utils.get_logger("DEBUG")


class InferenceRecipe:
    """
    Recipe for generating tokens from a dense Transformer-based LLM.

    Currently this recipe supports single-GPU generation only. Speculative
    decoding is not supported.

    For more details on how to use this recipe for generation, please see our
    tutorial: https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#generation

    For using this recipe with a quantized model, please the following section of
    the above tutorial:
    https://pytorch.org/torchtune/main/tutorials/e2e_flow.html#speeding-up-generation-using-quantization
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device = utils.get_device(device=cfg.device)
        self._dtype = utils.get_dtype(dtype=cfg.dtype)
        self._quantizer = config.instantiate(cfg.quantizer)
        self._quantization_mode = utils.get_quantizer_mode(self._quantizer)

        utils.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        checkpointer = config.instantiate(cfg.checkpointer)
        if self._quantization_mode is None:
            ckpt_dict = checkpointer.load_checkpoint()
        else:
            # weights_only needs to be False when loading a quantized model
            # currently loading a quantized model is only supported with the
            # FullModelTorchTuneCheckpointer
            ckpt_dict = checkpointer.load_checkpoint(weights_only=False)

        self._model = self._setup_model(
            model_cfg=cfg.model,
            model_state_dict=ckpt_dict[utils.MODEL_KEY],
        )
        self._tokenizer = config.instantiate(cfg.tokenizer)

    def _setup_model(
        self,
        model_cfg: DictConfig,
        model_state_dict: Dict[str, Any],
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(model_cfg)

        if self._quantization_mode is not None:
            model = self._quantizer.quantize(model)
            model = model.to(device=self._device, dtype=self._dtype)

        model.load_state_dict(model_state_dict)

        # Validate model was loaded in with the expected dtype.
        utils.validate_expected_param_dtype(model.named_parameters(), dtype=self._dtype)
        logger.info(f"Model is initialized with precision {self._dtype}.")

        # Ensure the cache is setup on the right device
        with self._device:
            model.setup_caches(max_batch_size=1, dtype=self._dtype)

        return model

    @torch.no_grad()
    def generate(self, cfg: DictConfig):
        print("Evaluating the model...", flush=True)
        data = LoadHDF5Data(cfg.data_path)
        results = pd.DataFrame(columns=['functionSource', 'isVulnerable', 'llmResult'])
        prompt_template = "Is the following function vulnerable? Please answer Yes or No. "

        for step in range(1000):
            code_snippet = data[step]['functionSource'].decode('utf-8')
            tokens = self._tokenizer.encode(prompt_template + code_snippet, add_bos=True, add_eos=False)
            prompt = torch.tensor(tokens, dtype=torch.int, device=self._device)

            custom_generate_next_token = None

            t0 = time.perf_counter()
            generated_tokens = utils.generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
                eos_id=self._tokenizer.eos_id,
                custom_generate_next_token=custom_generate_next_token,
            )
            t = time.perf_counter() - t0

            llmResult = self._tokenizer.decode(generated_tokens)

            results = results._append({'functionSource': code_snippet.encode("utf-8"), 'isVulnerable': data[step]['isVulnerable'],
                                       'llmResult': llmResult.lower().encode("utf-8")}, ignore_index=True)

        print("Saving data to csv...", flush=True)
        results.to_csv("results.csv", index=False)
        print("Evaluation complete.", flush=True)


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="InferenceRecipe", cfg=cfg)
    recipe = InferenceRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.generate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())


# utils


def getStringFromVulnList(cwe_119, cwe_120, cwe_469, cwe_476, cwe_other):
    if cwe_119:
        return 'CWE-119'
    elif cwe_120:
        return 'CWE-120'
    elif cwe_469:
        return 'CWE-469'
    elif cwe_476:
        return 'CWE-476'
    elif cwe_other:
        return 'CWE-other'
    else:
        return False


def LoadHDF5Data(data_path):
    print("Loading hdf5 data from %s..." % data_path)
    with h5py.File(data_path, 'r') as f:
        data = []
        cwe_119 = np.array(f['CWE-119'])
        cwe_120 = np.array(f['CWE-120'])
        cwe_469 = np.array(f['CWE-469'])
        cwe_476 = np.array(f['CWE-476'])
        cwe_other = np.array(f['CWE-other'])
        function_source = np.array(f['functionSource'])
        for i in range(len(function_source)):
            dataPoint = {
                'functionSource': function_source[i],
                'isVulnerable': getStringFromVulnList(cwe_119[i], cwe_120[i], cwe_469[i], cwe_476[i], cwe_other[i])
            }
            data.append(dataPoint)
    return data
