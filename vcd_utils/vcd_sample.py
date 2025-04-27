import copy
import inspect
import warnings
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput, GreedySearchOutput, BeamSearchOutput
from transformers.generation.beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .attention_adapter import LLaVaAttentionerManager
import torch.optim as optim
from .image_process import vis_mask, plot_distribution, build_neighbourhood_mask, vis_attention, vis_attn_sum


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    input_ids_cd: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores


    # output_attentions = (
    #     output_attentions if output_attentions is not None else self.generation_config.output_attentions
    # )
    # output_hidden_states = (
    #     output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    # )
    output_attentions = True
    output_hidden_states = True


    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    gamma = model_kwargs.get("use_mask")
    use_mask = model_kwargs.get("use_mask")
    mask_mode = model_kwargs.get("mask_mode")
    key_pos = model_kwargs.get("key_pos")
    use_key_pos = model_kwargs.get("use_key_pos")
    tokenizer = model_kwargs.get("tokenizer")
    model_kwargs_cd = model_kwargs.copy()
    flag = 0
    flag2 = 0
    attn_cd = None
    past_num = 1
    attn_sum = []
    if use_key_pos:
        flag = 1
    
    vis_flag = 1
    all_sums = []
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        with torch.no_grad():
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        with torch.inference_mode():
            with torch.no_grad():
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    key_pos = key_pos if flag == 1 else [],
                    modi_pos = True if mask_mode == "imccd" else False,
                    gamma = gamma
                )

        # TODO: statistic the attention activation
        # import pdb;pdb.set_trace()
        # if vis_flag:    
        #     sample_id = model_kwargs.get("sample_id")
        #     attn_last_layer = outputs['attentions'][-1]   # (bsz, num_head, seq_len, tgt_len)
            
        #     ind_start = key_pos[0]["image_token_start"].item()
        #     ind_end = (key_pos[0]["image_token_end"] + attn_last_layer.shape[3]).item()
            
        #     vis_attn = attn_last_layer[:, :, -1, ind_start: ind_end]    # (bsz, num_head, tgt_len)
        #     vis_attn = vis_attn.cpu().numpy()

        #     # save to disk
        #     save_dir = '/disk/work/jczhang/code/IMCCD/experiments/output/vis_attn/'
        #     np.save(f'{save_dir}/cross_modal_attn_last_layer_{sample_id}.npy', vis_attn)
        #     vis_flag = 0
        
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        if flag == 0 and key_pos != None:
            token_length = outputs['attentions'][-1].shape[3]
            flag = 1
            for i in range(len(key_pos)):
                key_pos[i]["image_token_end"] = key_pos[i]["image_token_end"] + token_length 
        if use_mask:
            if mask_mode == "imccd" or mask_mode == "ved":
                if attn_cd is None:
                    output_attns  = [attn for attn in outputs['attentions']]
                    output_attns = torch.cat(output_attns, dim=0)


                    try:
                        image_token_end = key_pos[0]['image_token_end']
                    except:
                        import pdb;pdb.set_trace()  
                    past_attns = output_attns[:, :, :image_token_end] 
                    past_num = output_attns.shape[2] - image_token_end 
                    model_kwargs_cd['past_key_values'] = [(keys[:, :, :image_token_end], values[:, :, :image_token_end]) for keys, values in outputs['past_key_values']]
                    model_kwargs_cd['past_key_values_without_pos'] = [(keys[:, :, :image_token_end:], values[:, :, :image_token_end]) for keys, values in outputs['past_key_values_without_pos']]
                else:
                    past_attns = attn_cd 
                    past_num = 1
            else:
                output_attns  = [abs(attn) for attn in outputs['attentions']]
                output_attns = torch.cat(output_attns, dim=0)
                output_attns = output_attns  / (output_attns .sum(dim=2) + 1e-7 * (output_attns .sum(dim=2) == 0).float() ).unsqueeze(2)
                output_attns  = output_attns.mean(dim = 0)
                past_attns = output_attns[-1]
        # if use_mask and use_kvcache:
        #     if mask_mode == "imccd" or mask_mode == "ved":
        #         if attn_cd is None:
        #             output_attns  = [attn for attn in outputs['attentions']]
        #             output_attns = torch.cat(output_attns, dim=0)
        #             try:
        #                 image_token_end = key_pos[0]['image_token_end']
        #             except:
        #                 import pdb;pdb.set_trace()  
        #             past_attns = output_attns[:, :, :image_token_end] 
        #             past_num = output_attns.shape[2] - image_token_end 
        #             model_kwargs_cd['past_key_values'] = [(keys[:, :, :image_token_end], values[:, :, :image_token_end]) for keys, values in outputs['past_key_values']]
        #             model_kwargs_cd['past_key_values_without_pos'] = [(keys[:, :, :image_token_end:], values[:, :, :image_token_end]) for keys, values in outputs['past_key_values_without_pos']]
        #         else:
        #             past_attns = attn_cd 
        #             past_num = 1
        #     else:
        #         output_attns  = [abs(attn) for attn in outputs['attentions']]
        #         output_attns = torch.cat(output_attns, dim=0)
        #         output_attns = output_attns  / (output_attns .sum(dim=2) + 1e-7 * (output_attns .sum(dim=2) == 0).float() ).unsqueeze(2)
        #         output_attns  = output_attns.mean(dim = 0)
        #         past_attns = output_attns[-1]

        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )

        if use_cd:
            with torch.inference_mode():
                with torch.no_grad():
                    ## cd_comments: forward pass of the model with distorted image input
                    if use_mask:
                        if model_kwargs_cd.get('images') != None:
                            model_kwargs_cd['images_cd'] = model_kwargs_cd['images']
                        elif model_kwargs_cd.get('inputs_embeds') != None:
                            model_kwargs_cd['images_cd'] = model_kwargs_cd['inputs_embeds']
                        else:
                            import pdb;pdb.set_trace()

                        # import pdb;pdb.set_trace()
                        model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, past_num = past_num, **model_kwargs_cd)
                        outputs_cd = self(
                            **model_inputs_cd,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                            past_attns = past_attns,
                            modi_mask = True,
                            mask_mode = mask_mode,
                            modi_pos = True if mask_mode == "imccd" else False,
                            key_pos = key_pos if flag == 1 else [],
                        )
                    else:
                        if input_ids_cd != None:
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids_cd, **model_kwargs_cd)
                            add_mask = torch.ones(input_ids.shape[0], input_ids_cd.shape[1]).to(input_ids.device)
                            model_inputs_cd['attention_mask'] = add_mask 
                        elif model_kwargs_cd.get('attention_mask_cd')  != None:
                            add_mask = torch.ones(input_ids.shape[0], input_ids.shape[1] - 1).to(input_ids.device)
                            model_kwargs_cd['attention_mask'] = torch.cat((model_kwargs_cd['attention_mask_cd'], add_mask ), dim = 1)
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)

                        else:
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)

                        outputs_cd = self(
                            **model_inputs_cd,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                        )        
                    attn_cd = [attn for attn in outputs_cd['attentions']]   
                    attn_cd = torch.cat(attn_cd, dim=0)   
                    tmp_output_attns = attn_cd.to(torch.float32)
                    next_token_logits_cd = outputs_cd.logits[:, -1, :]
                    
                    


                    ## cd_comments: pre-process logits from contrastive inputs
                    cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
   
                    cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
                    
                    use_entropy = False
                    check_word = False
                    if use_entropy:
                        next_prob_cd = nn.functional.softmax(next_token_logits_cd, dim=-1)
                        epsilon = 1e-6
                        entropy = -torch.sum(next_prob_cd * torch.log(next_prob_cd + epsilon), dim =1)
                        cd_alpha = 5 * entropy
                    if use_mask and tokenizer:
                        if check_word: 
                            next_logits = logits_processor(input_ids, next_token_logits)
                            next_logits= logits_warper(input_ids, next_logits)
                            next_probs = nn.functional.softmax(next_logits, dim=-1)
                            next_tokens = torch.multinomial(next_probs, num_samples=1).squeeze(1)
                            next_word = tokenizer.decode(next_tokens)
                            # tokenizer.batch_decode(input_ids)
                            doc = self.tagging(next_word)
                            # valid_list = ["NOUN", "PROPN", "ADD"] #, "ADJ"]
                            # import pdb;pdb.set_trace()
                            valid_list = ["NOUN", "PROPN"]
                            # valid_list = ["NOUN", "PROPN", "ADJ", "NUM", "VERB"]
                            if len(doc) < 1:
                                cd_alpha = 0
                            elif '.' in  next_word or ',' in next_word:
                                cd_alpha = 0
                            elif doc[0].pos_ not in valid_list:
                                cd_alpha = 0
                        else:
                            saliency_list  = [attn for attn in outputs['attentions']]
                            all_saliency = torch.cat(saliency_list, dim=0)
                            #attn_softmax = nn.functional.softmax(all_saliency, dim = 3)
                            attn_softmax = all_saliency 
                            attn_sum = attn_softmax[: ,: ,-1: , key_pos[0]['image_token_start']:key_pos[0]['image_token_end']].sum(dim=3).mean()
                            all_sums.append(attn_sum)

                            values = torch.tensor([t.item() for t in all_sums[:30]])

                            mean = torch.mean(values)
                            std = torch.std(values)
                            if attn_sum < mean:
                                cd_alpha = 0
                    # version 1  set cutoff for Adaptive Plausibility Constraints
                    # probs = nn.functional.softmax(next_token_logits, dim=-1)
                    # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

                    # version 2 set cutoff for Adaptive Plausibility Constraints
                    cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                    diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd


                    cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))


                    ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
                    cd_logits = logits_processor(input_ids, cd_logits)
                    cd_logits = logits_warper(input_ids, cd_logits)

                    next_token_scores = cd_logits
                    cd_probs = nn.functional.softmax(cd_logits, dim=-1)
                    next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)



        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if input_ids_cd != None:
            input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)
        
        
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)

        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids


def greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    input_ids_cd: Optional[torch.LongTensor] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    # logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores


    # output_attentions = (
    #     output_attentions if output_attentions is not None else self.generation_config.output_attentions
    # )
    # output_hidden_states = (
    #     output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    # )
    output_attentions = True
    output_hidden_states = True


    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    use_mask = model_kwargs.get("use_mask")
    mask_mode = model_kwargs.get("mask_mode")
    key_pos = model_kwargs.get("key_pos")
    use_key_pos = model_kwargs.get("use_key_pos")
    use_kvcache = model_kwargs.get("use_kvcache")       # whether to accelerate
    model_kwargs_cd = model_kwargs.copy()
    tokenizer = model_kwargs.get("tokenizer")
    flag = 0
    flag2 = 0
    attn_cd = None
    past_num = 1
    all_sums = []
    if use_key_pos:
        flag = 1

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break


        # prepare model inputs
        with torch.no_grad():
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        with torch.inference_mode():
            with torch.no_grad():
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    key_pos = key_pos if flag == 1 else [],
                    modi_pos = True if mask_mode == "imccd" else False,
                )
        
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need
        next_token_logits = outputs.logits[:, -1, :]
        if flag == 0 and key_pos != None:
            token_length = outputs['attentions'][-1].shape[3]
            flag = 1
            for i in range(len(key_pos)):
                key_pos[i]["image_token_end"] = key_pos[i]["image_token_end"] + token_length 
        if use_mask:
            if mask_mode == "imccd" or mask_mode == "ved":
                if attn_cd is None:
                    output_attns  = [attn for attn in outputs['attentions']]
                    output_attns = torch.cat(output_attns, dim=0)
                    try:
                        image_token_end = key_pos[0]['image_token_end']
                    except:
                        import pdb;pdb.set_trace()  
                    past_attns = output_attns[:, :, :image_token_end] 
                    past_num = output_attns.shape[2] - image_token_end 
                    model_kwargs_cd['past_key_values'] = [(keys[:, :, :image_token_end], values[:, :, :image_token_end]) for keys, values in outputs['past_key_values']]
                    model_kwargs_cd['past_key_values_without_pos'] = [(keys[:, :, :image_token_end:], values[:, :, :image_token_end]) for keys, values in outputs['past_key_values_without_pos']]
                else:
                    past_attns = attn_cd 
                    past_num = 1
            else:
                output_attns  = [abs(attn) for attn in outputs['attentions']]
                output_attns = torch.cat(output_attns, dim=0)
                output_attns = output_attns  / (output_attns .sum(dim=2) + 1e-7 * (output_attns .sum(dim=2) == 0).float() ).unsqueeze(2)
                output_attns  = output_attns.mean(dim = 0)
                past_attns = output_attns[-1]

        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        
        if use_cd:
            with torch.inference_mode():
                with torch.no_grad():
                    ## cd_comments: forward pass of the model with distorted image input
                    if use_mask:
                        if model_kwargs_cd.get('images') != None:
                            model_kwargs_cd['images_cd'] = model_kwargs_cd['images']
                        elif model_kwargs_cd.get('inputs_embeds') != None:
                            model_kwargs_cd['images_cd'] = model_kwargs_cd['inputs_embeds']
                        else:
                            import pdb;pdb.set_trace()

    
                        model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, past_num = past_num, **model_kwargs_cd)
                        outputs_cd = self(
                            **model_inputs_cd,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                            past_attns = past_attns,
                            modi_mask = True,
                            mask_mode = mask_mode,
                            modi_pos = True if mask_mode == "imccd" else False,
                            key_pos = key_pos if flag == 1 else [],
                        )
                    else:
                        if input_ids_cd != None:
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids_cd, **model_kwargs_cd)
                            add_mask = torch.ones(input_ids.shape[0], input_ids_cd.shape[1]).to(input_ids.device)
                            model_inputs_cd['attention_mask'] = add_mask 
                        elif model_kwargs_cd.get('attention_mask_cd')  != None:
                            add_mask = torch.ones(input_ids.shape[0], input_ids.shape[1] - 1).to(input_ids.device)
                            model_kwargs_cd['attention_mask'] = torch.cat((model_kwargs_cd['attention_mask_cd'], add_mask ), dim = 1)
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)

                        else:
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)

                        outputs_cd = self(
                            **model_inputs_cd,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                        )        
                    attn_cd = [attn for attn in outputs_cd['attentions']]   
                    attn_cd = torch.cat(attn_cd, dim=0)   
  
                    next_token_logits_cd = outputs_cd.logits[:, -1, :]

                    ## cd_comments: pre-process logits from contrastive inputs
                    cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
                    cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
                    check_word = False
                    if tokenizer and use_mask:
                        if check_word: 
                            next_logits = logits_processor(input_ids, next_token_logits)
                            next_logits= logits_warper(input_ids, next_logits)
                            next_probs = nn.functional.softmax(next_logits, dim=-1)
                            next_tokens = torch.multinomial(next_probs, num_samples=1).squeeze(1)
                            next_word = tokenizer.decode(next_tokens)
                            # tokenizer.batch_decode(input_ids)
                            doc = self.tagging(next_word)
                            # valid_list = ["NOUN", "PROPN", "ADD"] #, "ADJ"]
                            # import pdb;pdb.set_trace()
                            valid_list = ["NOUN", "PROPN"]
                            # valid_list = ["NOUN", "PROPN", "ADJ", "NUM", "VERB"]
                            if len(doc) < 1:
                                cd_alpha = 0
                            elif '.' in  next_word or ',' in next_word:
                                cd_alpha = 0
                            elif doc[0].pos_ not in valid_list:
                                cd_alpha = 0
                        else:
                            # import pdb;pdb.set_trace()
                            saliency_list  = [attn for attn in outputs['attentions']]
                            all_saliency = torch.cat(saliency_list, dim=0)
                            attn_softmax =  all_saliency
                            # attn_softmax = nn.functional.softmax(all_saliency, dim = 3)
                            attn_sum = attn_softmax[: , : ,-1: , key_pos[0]['image_token_start']:key_pos[0]['image_token_end']].sum(dim=3).mean()
                            all_sums.append(attn_sum)

                            values = torch.tensor([t.item() for t in all_sums][-30:])

                            mean = torch.mean(values)
                            std = torch.std(values)
                            if attn_sum < mean:
                                cd_alpha = 0
                            # if len(all_sums) > 100:
                            #     import pdb;pdb.set_trace()

                    # version 1  set cutoff for Adaptive Plausibility Constraints
                    # probs = nn.functional.softmax(next_token_logits, dim=-1)
                    # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

                    # version 2 set cutoff for Adaptive Plausibility Constraints
                    cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                    diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd

                    cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

                    ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
                    next_token_scores = logits_processor(input_ids, cd_logits)
                    next_tokens = torch.argmax(next_token_scores, dim=-1)
        else:
            next_tokens_scores = logits_processor(input_ids, next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)


        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if input_ids_cd != None:
            input_ids_cd = torch.cat([input_ids_cd, next_tokens[:, None]], dim=-1)
        
        
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)

        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids




def beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    input_ids_cd: Optional[torch.LongTensor] = None,
    **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
      
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores


    # output_attentions = (
    #     output_attentions if output_attentions is not None else self.generation_config.output_attentions
    # )
    # output_hidden_states = (
    #     output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    # )
    output_attentions = True
    output_hidden_states = True


    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # beam search initialization
    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
    # of the first beam are considered to avoid sampling the exact same tokens across all beams.
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view((batch_size * num_beams,))

    this_peer_finished = False  # used by synced_gpus only

    # IMCCD settings
    use_mask = model_kwargs.get("use_mask")
    mask_mode = model_kwargs.get("mask_mode")
    key_pos = model_kwargs.get("key_pos")
    use_key_pos = model_kwargs.get("use_key_pos")
    tokenizer = model_kwargs.get("tokenizer")
    model_kwargs_cd = model_kwargs.copy()
    flag = 0
    flag2 = 0
    attn_cd = None
    past_num = 1
    attn_sums = []
    all_sums = [] 
    if use_key_pos:
        flag = 1

    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break


        # prepare model inputs
        with torch.no_grad():
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        with torch.inference_mode():
            with torch.no_grad():
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    key_pos = key_pos if flag == 1 else [],
                    modi_pos = True if mask_mode == "imccd" else False,
                )
        
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        if flag == 0 and key_pos != None:
            token_length = outputs['attentions'][-1].shape[3]
            flag = 1
            for i in range(len(key_pos)):
                key_pos[i]["image_token_end"] = key_pos[i]["image_token_end"] + token_length 
        if use_mask:
            if mask_mode == "imccd" or mask_mode == "ved":
                if attn_cd is None:
                    output_attns  = [attn for attn in outputs['attentions']]
                    output_attns = torch.cat(output_attns, dim=0)
                    try:
                        image_token_end = key_pos[0]['image_token_end']
                    except:
                        import pdb;pdb.set_trace()  
                    past_attns = output_attns[:, :, :image_token_end] 
                    past_num = output_attns.shape[2] - image_token_end 
                    model_kwargs_cd['past_key_values'] = [(keys[:, :, :image_token_end], values[:, :, :image_token_end]) for keys, values in outputs['past_key_values']]
                    model_kwargs_cd['past_key_values_without_pos'] = [(keys[:, :, :image_token_end], values[:, :, :image_token_end]) for keys, values in outputs['past_key_values_without_pos']]
                else:
                    past_attns = attn_cd 
                    past_num = 1
            else:
                output_attns  = [abs(attn) for attn in outputs['attentions']]
                output_attns = torch.cat(output_attns, dim=0)
                output_attns = output_attns  / (output_attns .sum(dim=2) + 1e-7 * (output_attns .sum(dim=2) == 0).float() ).unsqueeze(2)
                output_attns  = output_attns.mean(dim = 0)
                past_attns = output_attns[-1]

        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        
        if use_cd:
            with torch.inference_mode():
                with torch.no_grad():
                    ## cd_comments: forward pass of the model with distorted image input
                    if use_mask:
                        if model_kwargs_cd.get('images') != None:
                            model_kwargs_cd['images_cd'] = model_kwargs_cd['images']
                        elif model_kwargs_cd.get('inputs_embeds') != None:
                            model_kwargs_cd['images_cd'] = model_kwargs_cd['inputs_embeds']
                        else:
                            import pdb;pdb.set_trace()

                        # import pdb;pdb.set_trace()
                        model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, past_num = past_num, **model_kwargs_cd)
                        outputs_cd = self(
                            **model_inputs_cd,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                            past_attns = past_attns,
                            modi_mask = True,
                            mask_mode = mask_mode,
                            modi_pos = True if mask_mode == "imccd" else False,
                            key_pos = key_pos if flag == 1 else [],
                        )
                    else:
                        if input_ids_cd != None:
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids_cd, **model_kwargs_cd)
                            add_mask = torch.ones(input_ids.shape[0], input_ids_cd.shape[1]).to(input_ids.device)
                            model_inputs_cd['attention_mask'] = add_mask 
                        elif model_kwargs_cd.get('attention_mask_cd')  != None:
                            add_mask = torch.ones(input_ids.shape[0], input_ids.shape[1] - 1).to(input_ids.device)
                            model_kwargs_cd['attention_mask'] = torch.cat((model_kwargs_cd['attention_mask_cd'], add_mask ), dim = 1)
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)

                        else:
                            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)

                        outputs_cd = self(
                            **model_inputs_cd,
                            return_dict=True,
                            output_attentions=output_attentions_wo_img,
                            output_hidden_states=output_hidden_states_wo_img,
                        )        
                    attn_cd = [attn for attn in outputs_cd['attentions']]   
                    attn_cd = torch.cat(attn_cd, dim=0)   
  
                    next_token_logits_cd = outputs_cd.logits[:, -1, :]
                    
                    


                    ## cd_comments: pre-process logits from contrastive inputs
                    cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5

                    

                    cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
                    check_word = False
                    if False and tokenizer:
                        if check_word: 
                            next_logits = logits_processor(input_ids, next_token_logits)
                            next_logits= logits_warper(input_ids, next_logits)
                            next_probs = nn.functional.softmax(next_logits, dim=-1)
                            next_tokens = torch.multinomial(next_probs, num_samples=1).squeeze(1)
                            next_word = tokenizer.decode(next_tokens)
                            # tokenizer.batch_decode(input_ids)
                            doc = self.tagging(next_word)
                            # valid_list = ["NOUN", "PROPN", "ADD"] #, "ADJ"]
                            # import pdb;pdb.set_trace()
                            valid_list = ["NOUN", "PROPN"]
                            # valid_list = ["NOUN", "PROPN", "ADJ", "NUM", "VERB"]
                            if len(doc) < 1:
                                cd_alpha = 0
                            elif '.' in  next_word or ',' in next_word:
                                cd_alpha = 0
                            elif doc[0].pos_ not in valid_list:
                                cd_alpha = 0
                        else:
                            saliency_list  = [attn for attn in outputs['attentions']]
                            all_saliency = torch.cat(saliency_list, dim=0)
                            attn_softmax = nn.functional.softmax(all_saliency, dim = 3)
                            attn_sum = attn_softmax[: ,: ,-1: , key_pos[0]['image_token_start']:key_pos[0]['image_token_end']].sum(dim=3).mean()
                            all_sums.append(attn_sum)

                            values = torch.tensor([t.item() for t in all_sums])

                            mean = torch.mean(values)
                            std = torch.std(values)
                            if attn_sum < mean :
                                cd_alpha = 0
                    # version 1  set cutoff for Adaptive Plausibility Constraints
                    # probs = nn.functional.softmax(next_token_logits, dim=-1)
                    # cutoff = cd_beta * probs.max(dim=-1, keepdim=True).values

                    # version 2 set cutoff for Adaptive Plausibility Constraints
                    cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
                    diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd

                    cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

                    next_token_logits = cd_logits

                    # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
                    # cannot be generated both before and after the `nn.functional.log_softmax` operation.
                    next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
                    next_token_scores = nn.functional.log_softmax(
                        next_token_logits, dim=-1
                    )  # (batch_size * num_beams, vocab_size)

                    next_token_scores_processed = logits_processor(input_ids, next_token_scores)
                    next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            
        else:
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)


        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores_processed,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # reshape for beam search
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
        )

        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size

        # stateless
        beam_outputs = beam_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            beam_indices=beam_indices,
        )

        beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        beam_idx = beam_outputs["next_beam_indices"]

        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
        
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)

        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if model_kwargs["past_key_values"] is not None:
            model_kwargs["past_key_values"] = self._reorder_cache(model_kwargs["past_key_values"], beam_idx)

        if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

        # increase cur_len
        cur_len = cur_len + 1

        if beam_scorer.is_done or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True
      
    sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )


    if return_dict_in_generate:
        if not output_scores:
            sequence_outputs["sequence_scores"] = None

        if self.config.is_encoder_decoder:
            return BeamSearchEncoderDecoderOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return BeamSearchDecoderOnlyOutput(
                sequences=sequence_outputs["sequences"],
                sequences_scores=sequence_outputs["sequence_scores"],
                scores=scores,
                beam_indices=sequence_outputs["beam_indices"],
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return sequence_outputs["sequences"]

def evolve_vcd_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.utils.GenerationMixin.greedy_search = greedy_search
    transformers.generation.utils.GenerationMixin.beam_search = beam_search
