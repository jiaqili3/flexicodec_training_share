from typing import Dict, Optional, Union
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy


class TransformerLM(torch.nn.Module):
    """
    TransformerLM Module
    """

    def __init__(
        self,
        text_encoder_input_size: int,
        llm_input_size: int,
        llm_output_size: int,
        text_token_size: int,
        speech_token_size: int,
        text_encoder: torch.nn.Module,
        llm: torch.nn.Module,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        spk_embed_dim: int = 192,
    ):
        """
        :param text_encoder_input_size:
        :param llm_input_size:
        :param llm_output_size:
        :param text_token_size:
        :param speech_token_size:
        :param text_encoder:
        :param llm:
        :param length_normalized_loss:
        :param lsm_weight:
        :param spk_embed_dim:
        """
        super().__init__()
        self.llm_input_size = llm_input_size
        self.speech_token_size = speech_token_size
        # 1. build text token inputs related modules
        self.text_embedding = torch.nn.Embedding(
            text_token_size, text_encoder_input_size
        )
        self.text_encoder = text_encoder
        self.text_encoder_affine_layer = nn.Linear(
            self.text_encoder.output_size(), llm_input_size
        )

        # 2. build speech token language model related modules
        self.sos_eos = 0
        self.task_id = 1
        self.llm_embedding = torch.nn.Embedding(2, llm_input_size)
        self.llm = llm
        self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(speech_token_size, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

    def encode(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ):
        """
        :param text:
        :param text_lengths:
        :return:
        """
        encoder_out, encoder_mask = self.text_encoder(
            text, text_lengths, decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        encoder_out = self.text_encoder_affine_layer(encoder_out)
        return encoder_out, encoder_out_lens

    def pad_unpad_sequence(
        self,
        sos_eos_emb,
        embedding,
        text_token,
        text_token_len,
        task_id_emb,
        speech_token,
        speech_token_len,
    ):
        """
        :param sos_eos_emb:
        :param embedding:
        :param text_token:
        :param text_token_len:
        :param task_id_emb:
        :param speech_token:
        :param speech_token_len:
        :return:
        """
        B = text_token.shape[0]
        text_token = unpad_sequence(text_token, text_token_len.cpu(), batch_first=True)
        speech_token = unpad_sequence(
            speech_token, speech_token_len.cpu(), batch_first=True
        )
        if embedding is None:
            embedding = torch.zeros(
                B, 1, self.llm_input_size, device=sos_eos_emb.device
            )
        lm_input = [
            torch.concat(
                [
                    sos_eos_emb.squeeze(dim=0),
                    embedding[i],
                    text_token[i],
                    task_id_emb.squeeze(dim=0),
                    speech_token[i],
                ],
                dim=0,
            )
            for i in range(len(text_token))
        ]
        lm_input_len = torch.tensor([i.size(0) for i in lm_input], dtype=torch.int32)
        lm_input = pad_sequence(lm_input, batch_first=True, padding_value=IGNORE_ID)
        return lm_input, lm_input_len

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            text_token: (B, L)
            text_token_lengths: (B,)
            speech_token: (B, T)
            speech_token_lengths: (B,)
            embedding: (B,)
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)

        if batch["embedding"] is not None:
            embedding = batch["embedding"].to(device)
        else:
            embedding = None

        # 1. prepare llm_target
        lm_target = [
            torch.tensor(
                [IGNORE_ID] * (2 + text_token_len[i])
                + speech_token[i, : speech_token_len[i]].tolist()
                + [self.speech_token_size]
            )
            for i in range(text_token.size(0))
        ]
        lm_target = pad_sequence(
            lm_target, batch_first=True, padding_value=IGNORE_ID
        ).to(device)

        # 2. encode text_token
        text_token = self.text_embedding(text_token)
        text_token, text_token_len = self.encode(text_token, text_token_len)

        # 3. embedding projection
        if embedding is not None:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(1)

        # 4. eos and task_id
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)

        # 5. encode speech_token
        speech_token = self.speech_embedding(speech_token)

        # 6. unpad and pad
        lm_input, lm_input_len = self.pad_unpad_sequence(
            sos_eos_emb,
            embedding,
            text_token,
            text_token_len,
            task_id_emb,
            speech_token,
            speech_token_len,
        )

        # 7. run lm forward
        lm_output, lm_output_mask = self.llm(lm_input, lm_input_len.to(device))
        logits = self.llm_decoder(lm_output)
        loss = self.criterion_ce(logits, lm_target)
        acc = th_accuracy(
            logits.view(-1, self.speech_token_size + 1),
            lm_target,
            ignore_label=IGNORE_ID,
        )
        return {"loss": loss, "acc": acc}

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        sampling: Union[bool, int, float] = True,
        beam_size: int = 1,
        ignore_eos: bool = True,
    ):
        """
        :param weighted_scores:
        :param sampling:
        :param beam_size:
        :param ignore_eos:
        :return:
        """
        while True:
            prob, indices = weighted_scores.softmax(dim=-1).topk(sampling)
            top_ids = prob.multinomial(beam_size, replacement=True)
            top_ids = indices[top_ids]
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
        return top_ids

    @torch.inference_mode()
    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        beam_size: int = 1,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> torch.Tensor:
        """
        :param text:
        :param text_len:
        :param prompt_text:
        :param prompt_text_len:
        :param prompt_speech_token:
        :param prompt_speech_token_len:
        :param embedding:
        :param beam_size:
        :param sampling:
        :param max_token_text_ratio:
        :param min_token_text_ratio:
        :return:
        """
        device = text.device
        if prompt_text is not None:
            text = torch.concat([prompt_text, text], dim=1)
            text_len += prompt_text_len
        if prompt_text_len is None:
            prompt_text_len = 0
        text = self.text_embedding(text)

        # 1. encode text
        text, text_len = self.encode(text, text_len)

        # 2. encode embedding
        if embedding is not None and embedding.shape[0] != 0:
            embedding = F.normalize(embedding, dim=1)
            embedding = self.spk_embed_affine_layer(embedding)
            embedding = embedding.unsqueeze(dim=1)
        else:
            embedding = torch.zeros(1, 0, self.llm_input_size).to(device)

        # 3. concat llm_input
        sos_eos_emb = self.llm_embedding.weight[self.sos_eos].reshape(1, 1, -1)
        task_id_emb = self.llm_embedding.weight[self.task_id].reshape(1, 1, -1)
        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size).to(device)
        lm_input = torch.concat(
            [sos_eos_emb, embedding, text, task_id_emb, prompt_speech_token_emb], dim=1
        )

        # 4. cal min/max_length
        min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
        max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

        # 5. step by step decode
        out_tokens = []
        offset = 0
        att_cache, cnn_cache = torch.zeros(
            (0, 0, 0, 0), device=lm_input.device
        ), torch.zeros((0, 0, 0, 0), device=lm_input.device)
        for i in range(max_len):
            y_pred, att_cache, cnn_cache = self.llm.forward_chunk(
                lm_input,
                offset=0,
                required_cache_size=-1,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                att_mask=torch.tril(
                    torch.ones(
                        (1, lm_input.shape[1], lm_input.shape[1]),
                        device=lm_input.device,
                    )
                ).to(torch.bool),
            )
            logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
            top_ids = self.sampling_ids(
                logp.squeeze(dim=0),
                sampling,
                beam_size,
                ignore_eos=True if i < min_len else False,
            ).item()
            if top_ids == self.speech_token_size:
                break
            out_tokens.append(top_ids)
            offset += lm_input.size(1)
            lm_input = self.speech_embedding.weight[top_ids].reshape(1, 1, -1)

        return torch.tensor([out_tokens], dtype=torch.int64, device=device)
