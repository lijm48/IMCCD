#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append(".") # Adds higher directory to python modules path.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithMask, CausalLMOutputWithPastIMCCD

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np




class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.cnt = 0
        self.cnt1 = 0
        self.cnt2 = 0
        self.attn_maps = None
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        past_key_values_without_pos: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        images_cd: Optional[torch.FloatTensor] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        past_attns: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        modi_mask: Optional[bool] = False,
        key_pos: Optional[list] = [],
        use_mask: Optional[bool] = False,
        mask_mode: Optional[str] = "",
        modi_pos: Optional[bool] = False,
        qs: Optional[str] = "",
        label: Optional[str] = "",
        input_ids_cd: torch.LongTensor = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # import pdb;pdb.set_trace()
        mask_sum = torch.zeros(1).to( images.device)
        with torch.no_grad():
            # import pdb;pdb.set_trace()
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, tmp_key_pos = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
            if len(key_pos) == 0:
                key_pos = tmp_key_pos

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # import pdb;pdb.set_trace()
        adaptive_mask = False
        tmp_past_attns = None
        if mask_mode == "ved":
            adaptive_mask = True
            tmp_past_attns = past_attns
        if mask_mode == "imccd":
            adaptive_mask = True
            tmp_past_attns = past_attns
            modi_pos = modi_pos
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_key_values_without_pos = past_key_values_without_pos,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            key_pos = key_pos, 
            adaptive_mask = adaptive_mask,
            past_attns = tmp_past_attns,
            modi_pos = modi_pos,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastIMCCD(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            past_key_values_without_pos=outputs.past_key_values_without_pos,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            # mask_sum=mask_sum,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, past_key_values_without_pos=None,  attention_mask=None, inputs_embeds=None, key_pos = [], use_mask = False, mask_mode = "",  **kwargs
    ):
        if use_mask and (mask_mode == 'gradient' or mask_mode == "adaptive_distribution_gradient" ) : #or mask_mode == "adaptive_pos"
            past_key_values = None
            past_key_values_without_pos=None
        if past_key_values:
            input_ids = input_ids[:, -1:]
            # import pdb;pdb.set_trace()

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "past_key_values_without_pos": past_key_values_without_pos,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                # "key_pos": key_pos
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_cd(
        self, input_ids, past_key_values=None, past_key_values_without_pos=None, attention_mask=None, inputs_embeds=None, key_pos_new = [], use_mask = False, past_num = 1, **kwargs
    ):
        # if use_mask:
        #     past_key_values = None
        if past_key_values:
            # import pdb;pdb.set_trace()
            input_ids = input_ids[:, -past_num:]
            # attention_mask = attention_mask[:, -past_num:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "past_key_values_without_pos": past_key_values_without_pos,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
                # "key_pos": key_pos_new
            }
        )
        return model_inputs

    def modify_attention_masks(
        self, attention_mask, saliency, key_pos
    ):
        try:
            image_token_start = key_pos[0]['image_token_start']
            image_token_end = key_pos[0]['image_token_end']
        except:
            import pdb;pdb.set_trace()
        # 
        saliency = saliency / (saliency[image_token_start:image_token_end].sum(dim=0) + 1e-7)
        # if self.cnt == 5:
        #     import pdb;pdb.set_trace()
        # saliency_mask = (saliency[image_token_start:image_token_end] < 0.002).float()
        saliency_mask = self.GMM_mask(saliency[image_token_start:image_token_end])
        mask_sum = saliency_mask.shape[0] - saliency_mask.sum()
        attention_mask[0, image_token_start:image_token_end] = attention_mask[0, image_token_start:image_token_end] * saliency_mask 
        # self.plot_distribution(saliency[image_token_start:image_token_end])
        return attention_mask, mask_sum 

        
            
    def GMM_mask(self, saliency):
        data = saliency.cpu().numpy()
        # 计算中位数
        median = np.median(data)

        # 计算中位绝对偏差（MAD）
        mad = np.median(np.abs(data - median))
        thres = max(median + mad, 0.0001)
        # import pdb;pdb.set_trace()
        mask = (saliency < thres).float()
        return mask 

    def plot_distribution(self,  saliency):
        plt.clf()
        plt.hist(saliency.cpu().numpy(), bins=100, edgecolor='black', range=(0, 0.02))
        plt.title('Data Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        self.cnt+=1
        # 保存图像
        plt.savefig('./imgs/attention_distribution'+str(self.cnt)+'.png')
        plt.clf()
        


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
