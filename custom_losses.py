# Copyright (c) Facebook, Inc. and its affiliates.
"""
Losses module contains implementations for various losses used generally
in vision and language space. One can register custom losses to be detected by
MMF using the following example.

.. code::

   from mmf.common.registry import registry
   from torch import nn


   @registry.register_loss("custom")
   class CustomLoss(nn.Module):
       ...

Then in your model's config you can specify ``losses`` attribute to use this loss
in the following way:

.. code::

   model_config:
       some_model:
           losses:
               - type: custom
               - params: {}
"""
import collections
import warnings
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmf.common.registry import registry
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence


class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_config`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (ListConfig): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instanttions of each loss
                                   passed in config
    """

    def __init__(self, loss_list):
        super().__init__()
        self.losses = nn.ModuleList()
        config = registry.get("config")
        self._evaluation_predict = False
        if config:
            self._evaluation_predict = config.get("evaluation", {}).get(
                "predict", False
            )

        for loss in loss_list:
            self.losses.append(MMFLoss(loss))

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        """Takes in the original ``SampleList`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample_list (SampleList): SampleList given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}
        if "targets" not in sample_list:
            if not self._evaluation_predict:
                warnings.warn(
                    "Sample list has not field 'targets', are you "
                    "sure that your ImDB has labels? you may have "
                    "wanted to run with evaluation.predict=true"
                )
            return output

        for loss in self.losses:
            output.update(loss(sample_list, model_output))

        if not torch.jit.is_scripting():
            registry_loss_key = "{}.{}.{}".format(
                "losses", sample_list["dataset_name"], sample_list["dataset_type"]
            )
            # Register the losses to registry
            registry.register(registry_loss_key, output)

        return output


class MMFLoss(nn.Module):
    """Internal MMF helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/vqa2/logit_bce": 27.4}``, in case
    `logit_bce` is used and SampleList is from `val` set of dataset `vqa2`.

    Args:
        params (type): Description of parameter `params`.

    .. note::

        Since, ``MMFLoss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = {}

        is_mapping = isinstance(params, collections.abc.MutableMapping)

        if is_mapping:
            if "type" not in params:
                raise ValueError(
                    "Parameters to loss must have 'type' field to"
                    "specify type of loss to instantiate"
                )
            else:
                loss_name = params["type"]
        else:
            assert isinstance(
                params, str
            ), "loss must be a string or dictionary with 'type' key"
            loss_name = params

        self.name = loss_name

        loss_class = registry.get_loss_class(loss_name)

        if loss_class is None:
            raise ValueError(f"No loss named {loss_name} is registered to registry")
        # Special case of multi as it requires an array
        if loss_name == "multi":
            assert is_mapping
            self.loss_criterion = loss_class(params)
        else:
            if is_mapping:
                loss_params = params.get("params", {})
            else:
                loss_params = {}
            self.loss_criterion = loss_class(**loss_params)

    def forward(self, sample_list: Dict[str, Tensor], model_output: Dict[str, Tensor]):
        loss = self.loss_criterion(sample_list, model_output)

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)

        if loss.dim() == 0:
            loss = loss.view(1)

        if not torch.jit.is_scripting():
            key = "{}/{}/{}".format(
                sample_list.dataset_type, sample_list.dataset_name, self.name
            )
        else:
            key = f"{self.name}"
        return {key: loss}

@registry.register_loss("mrl_custom")
class MarginRankingLossCustom(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        inp1 = scores[:,0]
        inp2 = scores[:,1]
        y = []
        for x in targets:
          if x==0: y.append(-1)
          else: y.append(1)
        y = torch.tensor(y)
        y = y.to("cuda:0")
        #print(torch.numel(scores), torch.numel(targets))
        #print(scores)
        #print(targets)
        loss = F.margin_ranking_loss(inp1, inp2, y, reduction="mean")
        return loss

@registry.register_loss("mlsml_custom")
class MultiLabelSoftMarginLossCust(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        #print(torch.numel(scores), torch.numel(targets))
        #print(scores)
        #print(targets)
        loss = F.multilabel_soft_margin_loss(scores, targets, reduction="mean")
        return loss

@registry.register_loss("focal_loss_custom")
class FocalLossCust(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        gamma = 2.

        #print(torch.numel(scores), torch.numel(targets))
        #print(scores)
        #print(targets)
        logprobs = F.log_softmax(scores, dim=-1)
        probs = torch.exp(logprobs)
        scores1 = ((1 - probs) ** gamma) * logprobs
        loss = F.nll_loss(scores1, targets )
        return loss

@registry.register_loss("focal_loss_custom1")
class FocalLossCust1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        import numpy as np
        scores = model_output["scores"]
        targets = sample_list["targets"]
        gamma = 2.

        weight =  [4446, 2895]
        inv_weights = [1/x for x in weight]
        sum_invs = sum(inv_weights)
        inv_weights = [(len(weight)*x)/sum_invs for x in inv_weights]
        inv_weights = torch.Tensor(inv_weights)

        #pt = torch.softmax(input, dim=1)
        #inp1 = (1-pt)**gamma
        #inp2 = torch.log(pt)
        #inp3 = inp1 * inp2
        logprobs = F.log_softmax(scores, dim=-1)
        probs = torch.exp(logprobs)
        inp3 = ((1 - probs) ** gamma) * logprobs
        N = len(scores)
        inp4 = inp3[np.arange(N),targets]
        wts = []
        for y in targets:
            wts.append(inv_weights[y])
        wts = torch.Tensor(wts)
        wts = wts.to("cuda:0")
        op = - torch.dot(inp4,wts)
        loss = op/N
        #loss = F.nll_loss(scores1, targets )
        return loss


@registry.register_loss("focal_loss_custom2")
class FocalLossCust2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        import numpy as np
        scores = model_output["scores"]
        targets = sample_list["targets"]
        gamma = 2.

        weight =  [2895, 4446]
        inv_weights = [1/x for x in weight]
        sum_invs = sum(inv_weights)
        inv_weights = [(len(weight)*x)/sum_invs for x in inv_weights]
        inv_weights = torch.Tensor(inv_weights)

        #pt = torch.softmax(input, dim=1)
        #inp1 = (1-pt)**gamma
        #inp2 = torch.log(pt)
        #inp3 = inp1 * inp2
        logprobs = F.log_softmax(scores, dim=-1)
        probs = torch.exp(logprobs)
        inp3 = ((1 - probs) ** gamma) * logprobs
        N = len(scores)
        inp4 = inp3[np.arange(N),targets]
        wts = []
        for y in targets:
            wts.append(inv_weights[y])
        wts = torch.Tensor(wts)
        wts = wts.to("cuda:0")
        op = - torch.dot(inp4,wts)
        loss = op/N
        #loss = F.nll_loss(scores1, targets )
        return loss
