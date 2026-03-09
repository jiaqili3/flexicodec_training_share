from typing import Dict, Optional, Union
import torch
import time
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, unpad_sequence

from cosyvoice.utils.common import IGNORE_ID
from cosyvoice.transformer.label_smoothing_loss import LabelSmoothingLoss
from cosyvoice.utils.common import th_accuracy
from einops import rearrange

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
        fast_llm: torch.nn.Module,
        fast_llm_output_size: int,
        length_normalized_loss: bool = True,
        lsm_weight: float = 0.0,
        spk_embed_dim: int = 192,
        num_heads: int = 2,
        depformer_multi_linear=0,
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
        self.fast_llm = fast_llm
        self.num_heads = num_heads

        # self.llm_decoder = nn.Linear(llm_output_size, speech_token_size + 1)
        self.criterion_ce = LabelSmoothingLoss(
            size=speech_token_size + 1,
            padding_idx=IGNORE_ID,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )

        # 3. [Optional] build speech token related modules
        self.speech_embedding = torch.nn.Embedding(
            speech_token_size + 1, llm_input_size
        )
        self.proj_into_group = nn.Linear(llm_input_size*self.num_heads, llm_input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, llm_input_size)

        self.depformer_multi_linear = depformer_multi_linear
        if depformer_multi_linear:
            assert isinstance(depformer_multi_linear, int)
            self.depformer_in = nn.ModuleList(
                [nn.Linear(llm_input_size, llm_input_size) for _ in range(depformer_multi_linear)]
            )
        # fast AR LLM
        self.llm_decoder = nn.Linear(fast_llm_output_size, speech_token_size + 1)


        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

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
        pad_eos=False,
        num_eos=0,
        eos_embedding=None,
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
        if pad_eos:
            for i in range(len(speech_token)):
                speech_token[i] = torch.cat([speech_token[i]]+num_eos*[eos_embedding.reshape(1,-1)], dim=0)
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
        """
        text_token = batch["text_token"].to(device)
        text_token_len = batch["text_token_len"].to(device)
        speech_token = batch["speech_token"].to(device)
        speech_token_len = batch["speech_token_len"].to(device)

        # unpad speech token to make it divisible by num_heads
        assert speech_token.shape[-1] % self.num_heads == 0

        if batch["embedding"] is not None:
            embedding = batch["embedding"].to(device)
        else:
            embedding = None

        # 1. prepare llm_target
        lm_target = [
            torch.tensor(
                [IGNORE_ID] * (2 + text_token_len[i]) * self.num_heads
                + speech_token[i, : speech_token_len[i]].tolist()
                + [self.speech_token_size] * self.num_heads
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

        # speech_embedding_clone: embedding of each speech token
        speech_embedding_clone = speech_token.clone().detach()

        # chunk the speech token
        speech_token = rearrange(speech_token, "b (t k) h -> b t (k h)", k=self.num_heads)
        speech_token = self.proj_into_group(speech_token)
        # B, T, K, H = speech_token.shape
        # speech_token = speech_token.view(B, T, H)
        assert (speech_token_len % self.num_heads == 0).all()
        speech_token_len = speech_token_len // self.num_heads
        # eos_embedding = self.speech_embedding(torch.tensor(self.speech_token_size, device=speech_token.device))

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
        B, T, _ = lm_output.shape
        lm_output = rearrange(lm_output, 'b t h -> (b t) 1 h')

        if self.depformer_multi_linear:
            fast_inputs = [self.depformer_in[0](lm_output)]
        else:
            fast_inputs = [lm_output]
        speech_embedding_clone = rearrange(speech_embedding_clone, 'b (t k) h -> b t k h', k=self.num_heads)

        for i in range(self.num_heads - 1):
            # the second input to fast layer
            fast_input_tmp = self.pad_unpad_sequence(
                sos_eos_emb,
                embedding,
                text_token,
                text_token_len,
                task_id_emb,
                speech_embedding_clone[:, :, i],
                speech_token_len,
            )[0] # [b, t, h]
            # # Drop the last token and rotate left, and pad eos embedding
            # fast_input_tmp = fast_input_tmp[:, 1:]
            # fast_input_tmp = torch.cat([fast_input_tmp, self.speech_embedding(torch.tensor(self.speech_token_size))])

            fast_input_tmp = rearrange(fast_input_tmp, 'b t h -> (b t) 1 h')
            
            if self.depformer_multi_linear:
                fast_inputs.append(self.depformer_in[i+1](lm_output) + fast_input_tmp)
            else:
                fast_inputs.append(fast_input_tmp)
        fast_inputs = torch.cat(fast_inputs, dim=1) # (b t) k h
        lm_output, _ = self.fast_llm(fast_inputs, self.num_heads*torch.ones(B*T, device=device, dtype=torch.int32))
        logits = self.llm_decoder(lm_output)
        logits_tmp = rearrange(logits, '(b t) k h -> b (t k) h', b=B)

        loss = self.criterion_ce(logits_tmp, lm_target)

        with torch.no_grad():
            # Reshape logits to keep `k` separate
            logits = rearrange(logits, '(b t) k h -> b t k h', b=B)

            # Reshape lm_target to match the logits structure
            lm_target = rearrange(lm_target, 'b (t k) -> b t k', k=self.num_heads)

            # Initialize a dictionary to store accuracy for each `k`
            accuracy_dict = {}

            # Iterate over each `k` and compute the accuracy separately
            for i in range(logits.size(2)):  # Iterate over dimension `k`
                logits_k = logits[:, :, i, :]  # Extract logits for the i-th dimension of `k`
                lm_target_k = lm_target[:, :, i]  # Extract the corresponding target for the i-th dimension of `k`
                
                # Get the predicted class (argmax) from the logits
                preds_k = torch.argmax(logits_k, dim=-1)  # Shape: [b, t]
                
                # Mask to ignore `IGNORE_ID` in target
                valid_mask = (lm_target_k != IGNORE_ID)
                
                # Compare predictions to the target
                correct_preds = (preds_k == lm_target_k) & valid_mask
                
                # Calculate accuracy: correct predictions / total valid predictions
                acc_k = correct_preds.sum().item() / valid_mask.sum().item()
                
                # Store the accuracy for this `k` in the dictionary
                accuracy_dict[f"acc_{i}"] = torch.tensor(acc_k)


        # logits = rearrange(logits, 'b t h -> (b t) h')
        # lm_target = rearrange(lm_target, 'b t -> (b t)')
        return {"loss": loss, "acc": accuracy_dict}

    def sampling_ids(
        self,
        weighted_scores: torch.Tensor,
        sampling: Union[bool, int, float] = True,
        beam_size: int = 1,
        ignore_eos: bool = True,
        sample_idx=None
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
            if sample_idx is None:
                top_ids = prob.multinomial(beam_size, replacement=True)
                ret_sample_idx = top_ids.item()
                top_ids = indices[top_ids]
            else:
                top_ids = indices[sample_idx]
                ret_sample_idx = None
            if (not ignore_eos) or (self.speech_token_size not in top_ids):
                break
            else:
                sample_idx = None
        return top_ids, ret_sample_idx

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
        min_token_text_ratio: float = 1,
        second_sampling = 2,
        require_all_stop_codes=False,
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
        # time1 = time.time()

        self.eval()
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

        # trim speech token to make it divisible by num_heads
        while prompt_speech_token.shape[-1] % self.num_heads != 0:
            prompt_speech_token = prompt_speech_token[..., 1:]
            prompt_speech_token_len -= 1

        if prompt_speech_token_len != 0:
            prompt_speech_token_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_token_emb = torch.zeros(1, 0, self.llm_input_size).to(device)

        # chunk the speech token
        # pad eos
        prompt_speech_token_emb = rearrange(
            prompt_speech_token_emb, "b (t k) h -> b t (k h)", k=self.num_heads
        )
        prompt_speech_token_emb = self.proj_into_group(prompt_speech_token_emb)
        prompt_speech_token_len = prompt_speech_token_len // self.num_heads

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

            y_pred_ori = y_pred[:, -1, None]
            # fast transformer
            fast_att_cache, fast_cnn_cache = torch.zeros(
                (0, 0, 0, 0), device=lm_input.device
            ), torch.zeros((0, 0, 0, 0), device=lm_input.device)

            fast_preds = []

            if self.depformer_multi_linear:
                y_pred = self.depformer_in[0](y_pred_ori)
            else:
                y_pred = y_pred_ori

            for j in range(self.num_heads):
                y_pred, fast_att_cache, fast_cnn_cache = self.fast_llm.forward_chunk(
                    y_pred,
                    offset=0,
                    required_cache_size=-1,
                    att_cache=fast_att_cache,
                    cnn_cache=fast_cnn_cache,
                    att_mask=torch.tril(
                        torch.ones(
                            (1, y_pred.shape[1], y_pred.shape[1]),
                            device=y_pred.device,
                        )
                    ).to(torch.bool),
                )
                logp = self.llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)
                top_ids, last_sample_idx = self.sampling_ids(
                    logp.squeeze(dim=0),
                    sampling=sampling,
                    beam_size=beam_size,
                    ignore_eos=True if (i < 10 or j != 0) else False,
                    # ignore_eos=True if (j != 0) else False,
                )
                if top_ids == self.speech_token_size:
                    if not require_all_stop_codes:
                        return torch.tensor([out_tokens], dtype=torch.int64, device=device)
                    else:
                        if j == self.num_heads - 1:
                            out_tokens.append(top_ids.item())
                            return torch.tensor([out_tokens], dtype=torch.int64, device=device)
                        
                # if len(out_tokens) == 10:
                #     time2 = time.time()
                #     print(f"10 token ar time: {time2 - time1}")
                #     return torch.tensor([out_tokens], dtype=torch.int64, device=device)

                out_tokens.append(top_ids.item())
                fast_preds.append(self.speech_embedding(top_ids))
                if j == self.num_heads - 1:
                    continue
                if self.depformer_multi_linear:
                    y_pred = self.speech_embedding(top_ids).reshape(1,1,-1) + self.depformer_in[j+1](y_pred_ori)
                else:
                    y_pred = self.speech_embedding(top_ids).reshape(1,1,-1)

            fast_preds = torch.cat(fast_preds, dim=-1)
            fast_preds = self.proj_into_group(fast_preds).reshape(1,1,-1)
            lm_input = fast_preds
        return torch.tensor([out_tokens], dtype=torch.int64, device=device)
