"""Implementation of the base model class."""
import abc
import logging
from typing import Optional, List

import torch
from torch import nn

def load_model_checkpoint_with_map(
    model, checkpoint_path: str, weight_map: List[List[str]], strict: bool = True
):
    """Load weights from a checkpoint into a model according to an weight initialization map.

    Model nodes are mapped to "model.*" in the destination name.
    If the checkpoint contains a "model_state_dict" key, its contents are mapped to "model.*" in
    the source name.

    Args:
        model: pytorch model module object
        checkpoint_path (str): checkpoint file path
        weight_map (List[List[str]]): a list of lists of 2 strings (source, destination). source is
            the name of the node inside the checkpoint, destination is the name of the model node.
        strict (bool): whether to enforce strict loading for each (source, destination) pair;
            See load_model_checkpoint. (default: True)
    """
    # load checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    LOG.info("Loading checkpoint %r", checkpoint_path)
    if "model_state_dict" in state_dict:
        new_state_dict = OrderedDict()
        for k, v in state_dict["model_state_dict"].items():
            new_state_dict[f"model.{k}"] = v
        state_dict = new_state_dict

    for entry in weight_map:
        if len(entry) != 2 or not all(isinstance(x, str) for x in entry):
            raise ValueError("weight_map must be a list of [source: str, dest:str]")
        source, dest = entry
        if not source or not dest:
            raise ValueError("source, dest cannot be empty")
        # find node in model
        dest_node = get_module_node({"model": model}, dest)
        source_dict = get_state_dict_node(state_dict, source)
        if not source_dict:
            LOG.warning(
                "Node %r not found in checkpoint %r, nothing will be loaded",
                source,
                checkpoint_path,
            )
        load_model_state_dict(dest_node, source_dict, strict=strict)




def load_model_checkpoint(model, checkpoint_path, strict: bool = True):
    """Loading the model state dict from the checkpoint path.

    Args:
        model: any CSDs model.
        checkpoint_path (str | dict): path where the model should be loaded from.
    """
    # For a model checkpoint from model registry, checkpoint_path will be a dict
    # with the model name, version, registry name, checkpoint number and filename.
    temp_dir = None
    if isinstance(checkpoint_path, dict) and checkpoint_path.get("model_name"):
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        checkpoint_path = ModelRegistryManager().download_model_checkpoint(
            checkpoint_path, temp_dir.name
        )

    if "moe" in dist.DIST_CONFIG:
        from CSDs.utils.dist import moe

        moe.load_model_chkp(model, checkpoint_path)
        return

    LOG.info("Loading model from %s", checkpoint_path)
    
    # Try loading with safetensors first, fall back to torch if it fails
    try:
        if checkpoint_path.endswith('.safetensors'):
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        LOG.warning(f"Failed to load with safetensors, falling back to torch: {e}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # even though it's not needed in latest code, let's still strip "module."
    # from name to provide backward-compatibility with old checkpoints that
    # were saved with model.state_dict() instead of model.module.state_dict()
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    if temp_dir is not None:
        temp_dir.cleanup()
    try:
        # Temporary renaming for retro-compatibility with TorchSpeech
        # All this hack will be removed in next version
        state_dict = ts_retrocomp_state_dict(state_dict)
        load_model_state_dict(model, state_dict, strict=strict)
    except (RuntimeError, KeyError):
        LOG.error("Problem in loading state_dict")
        raise



LOG = logging.getLogger(__name__)


class BaseModel(abc.ABC, nn.Module):
    """
    Base model class that provides unified interface to all CSDs models.

    All of the models have to inherit from this class and implement all
    abstract methods.

    .. note::
       All of the derived models should **not** implement their own "forward"
       method, but instead should implement
       "training_forward"/"inference_forward", depending on the scenario.

    Args:
        mode (str, optional): could be "train" or "infer". Used to distuingish
            if model is created for training or for inference (i.e., for
            conversion to ONNX or libtorch). This argument is read-only and
            should not be changed after model was created (as this can create
            inconsistency with any "inner" models, e.g. for teacher-student or
            ensemble cases). Defaults to "train".
        frozen_layers (list, optional): Whether to freeze specific weights of
            the model. For example, if
            ``frozen_layers: ['blstm_layers.0.', 'blstm_layers.1.']`` will
            freeze the first and second LSTM layers. In details, it will set
            requires_grad for parameter to False if any of the strings in
            frozen_layers is present in parameter name. Defaults to None.
        init_scale (float, optional): a scalar multiplier applied to all model
            weights (right after initialization). Mainly useful for
            mu-parametrization. Defaults to 1.0.
    """

    def __init__(self, mode="train", frozen_layers: Optional[List] = None, init_scale: float = 1.0):
        super().__init__()
        # mode is read-only for now. It's not really required,
        # but convenient to not worry about that in composite models.
        self._mode = mode
        self.frozen_layers = frozen_layers
        self.init_scale = init_scale
        self.trainer_callbacks = None

    @property
    def mode(self):
        """Mode of the model ("train" or "infer")."""
        return self._mode

    def layer_freeze(self, frozen_layers: list):
        """Will freeze certain model layers.

        Args:
            frozen_layers(list[str]): will use this list to set certain
                parameters requires_grad to False. In details, will do it
                if any of the strings in frozen_layers is present in parameter
                name.
        """
        frozen_layers = set(frozen_layers)

        for s, param in self.named_parameters():
            for this_layers in frozen_layers:
                if s.find(this_layers) >= 0:
                    param.requires_grad = False

    @abc.abstractmethod
    def training_forward(self, dl_output):
        """forward method implementation for the "train" mode.

        This method has a very flexible interface, so that it can be used
        interchangeably with different dataloaders and loss functions, that
        possibly define different parameters. Input and output of this method
        are dictionaries that can follow arbitrary format. The only requirement
        is that this format has to be compatible with any dataloader/loss/error
        function that can be used with this model. Dataloader output will be
        directly provided as the input to this method and this method's output
        will be directly provided as an input to the loss/error functions.

        .. note::
           You can also use CSDs.metrics.MetricsLogger class here to log
           any additional model-specific metrics (e.g. outputs of intermediate
           layers).

        Args:
            dl_output (dict): dataloader output dictionary, as is, without
                any modifications.

        Returns:
            dict: dictionary of outputs, in any format, but compatible with
            loss functions.

            For most of the loss functions, you should at least define
            an "output" key to specify model output.
        """

    def init_model(
        self,
        checkpoint=None,
        strict=True,
        init_map: Optional[List] = None,
        show_initial_weights=False,
    ):
        """The method can be used to initialize the model parameters.

        This base-class method provides implementation for loading pre-trained
        checkpoint weights. You should extend this method for any other
        model-specific initialization behavior that you want to define
        (users will be able to directly leverage this by providing additional
        parameters in the "model/init_model" config section).

        In additional to loading checkpoints, this function will set certain
        parameters to be non-trainable, depending on self.frozen_layers
        variable.

        Args:
            checkpoint (str | dict, optional): path to the checkpoint that should
                be loaded. Defaults to None (meaning all parameters will be
                randomly initialized).
            init_map (list, optional): list of weight initialization maps as described in
                `CSDs.utils.checkpoints.load_model_checkpoint_with_map`
            show_initial_weights (bool, optional): show minimum, average and maximum value of each
                parameter after initialization.
        """
        if self.frozen_layers is not None:
            self.layer_freeze(self.frozen_layers)

        if checkpoint is None:
            # if we do not load weights from checkpoint, we should apply
            # init_scale multiplication here.
            for pm in self.parameters():
                pm.data *= self.init_scale
        else:
            load_model_checkpoint(self, checkpoint, strict=strict)

        if init_map is not None:
            for entry in init_map:
                load_model_checkpoint_with_map(
                    model=self,
                    checkpoint_path=entry["checkpoint"],
                    weight_map=entry["map"],
                    strict=strict,
                )
        if show_initial_weights:
            LOG.info("Initial weights values: name | min | avg | max")
            for name, pm in self.named_parameters():
                vmax, vmin, vmean = pm.data.max(), pm.data.min(), pm.data.mean()
                LOG.info("    %s | %.2f | %.2f | %.2f", name, vmin, vmean, vmax)
            LOG.info("")

    def get_device(self):
        """
        Returns the device by querying the first element of `self.parameters`.

        Note that it should usually not be called inside the model ``__init__``
        method, because parameters might not have been created yet.

        Returns:
            torch.device: device of the current model
        """
        return next(self.parameters()).device

    def inference_forward(self, *args, **kwargs):
        """forward method implementation for the "infer" mode.

        This method is not implemented in the base model, so any derived model
        that has to support inference needs to define it.

        This function is used to convert the model to ONNX or libtorch, so that
        it can later be used in the inference engine. Therefore, it is important
        to make sure this method is clean and optimized (e.g. does not do any
        logging). You will also have to specify exact parameter names and their
        types in the signature of this function.

        .. note::
           When exporting model to ONNX or libtorch, we will first convert it
           to TorchScript intermediate representation, which places certain
           restrictions on what operations you can use inside this function.
           Refer to the
           `TorchScript docs <https://pytorch.org/docs/stable/jit.html>`_
           for more details.
        """
        raise NotImplementedError("Implement this function to support inference")

    @torch.jit.ignore
    def forward(self, *args, **kwargs):
        """Calls "training_forward" or "inference_forward", based on the mode.

        We need to make sure all models are called through this "forward"
        method, since PyTorch places certain hooks on it (e.g. for registering
        model parameters in distributed jobs). So it is important that all
        calls to the model are made with ``model.forward(...)`` or simply
        ``model(...)``, instead of directly calling ".training_forward" or
        ".inference_forward".
        """
        if self.mode == "train":
            return self.training_forward(*args, **kwargs)
        if self.mode == "infer":
            return self.inference_forward(*args, **kwargs)
        raise RuntimeError(f"Unknown mode {self.mode}")
