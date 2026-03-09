# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from tqdm import tqdm
import pickle
import librosa
import numpy as np


class CodecDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        assert isinstance(dataset, str)

        self.cfg = cfg

        # the path of the processed data
        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        # the name of the meta file, for example: "valid.json" and "train.json"
        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        # the path of the meta file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)

        # the metadata of your data, which is a list of dict
        # for example: [{"Uid": "61-70968-0060", "num_frames": 160000, "text": ..., "path": ...}]
        # uid is the unique identifier of the speech (e.g. the file name of the speech),
        # num_frames is the number of frames of the speech,
        # text is the text of the speech,
        # path is the path of the speech
        # you can change the content of the metadata according to your data
        self.metadata = self.get_metadata()

        # # the sorted list of speech index according to the number of frames, which is used for bucketing
        # self.all_num_frames = []
        # for i in range(len(self.metadata)):
        #     self.all_num_frames.append(self.metadata[i]["num_frames"])
        # self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        # self.num_frame_indices = np.array(
        #     sorted(
        #         range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
        #     )
        # )

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("metadata len: ", len(metadata))

        return metadata

    def random_crop(self, speech, max_length):
        if len(speech) <= max_length:
            # padding
            speech = np.pad(speech, (0, max_length - len(speech)), "constant")
            return speech

        start = random.randint(0, len(speech) - max_length)
        return speech[start : start + max_length]

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        single_feature = dict()

        # load speech
        speech, _ = librosa.load(utt_info["path"], sr=self.cfg.preprocess.sample_rate)
        speech = self.random_crop(speech, self.cfg.preprocess.max_length)

        single_feature["speech"] = speech

        return single_feature

    # def get_num_frames(self, index):
    #     utt_info = self.metadata[index]
    #     return utt_info["num_frames"]


class CodecCollator(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # speech
        packed_batch_features["speech"] = pad_sequence(
            [torch.tensor(utt["speech"]).float() for utt in batch], batch_first=True
        )

        return packed_batch_features


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
