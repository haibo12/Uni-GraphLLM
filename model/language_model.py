import torch
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class InstructGLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids= None,
        attention_mask= None,
        position_ids= None,
        past_key_values= None,
        inputs_embeds= None,
        labels= None,
        use_cache= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
        **kwargs
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def g_step(self, in_embeds, attention_mask, **kwargs):
        # Inputs are numerical inputs_embeds: first_model already spliced graph tokens in.
        return graph_g_step(self, in_embeds, attention_mask, **kwargs)


def graph_g_step(
    model,
    in_embeds,
    attention_mask,
    return_step_logits=False,
    yes_token_id=None,
    no_token_id=None,
    auc_source='logits',
    greedy=False,
    top_p=None,
):
    """Inference-time generation / Yes-No scoring shared by all backbones.

    With ``return_step_logits`` returns ``(ids, step_logits)``; ``step_logits`` is
    ``[N, 2]`` = (No, Yes) for ``auc_source='logits'``, and ``[N, 4]`` =
    (No, Yes) processed scores + (No, Yes) raw logits for ``'gen_scores'``.
    """
    model.eval()

    if greedy:
        sample_kwargs = dict(do_sample=False)
    else:
        # top_p=None keeps the backbone's generation_config value.
        sample_kwargs = dict(do_sample=True, temperature=0.7)
        if top_p is not None:
            sample_kwargs['top_p'] = top_p
    gen_kwargs = dict(
        inputs_embeds=in_embeds,
        attention_mask=attention_mask,
        max_new_tokens=80,
        **sample_kwargs,
    )

    if not return_step_logits:
        return model.generate(**gen_kwargs)

    if yes_token_id is None or no_token_id is None:
        raise ValueError("yes_token_id/no_token_id are required when return_step_logits=True.")

    if auc_source == 'gen_scores':
        # output.scores are post temperature/top-k/top-p: tokens outside the top-p
        # nucleus are -inf, so many samples get a hard 0/1 Yes probability.
        out = model.generate(**gen_kwargs, return_dict_in_generate=True,
                             output_scores=True, output_logits=True)
        scores = out.scores[0].float()
        raw = out.logits[0].float()
        step_logits = torch.stack(
            [scores[:, no_token_id], scores[:, yes_token_id],
             raw[:, no_token_id], raw[:, yes_token_id]],
            dim=-1,
        )
        return out.sequences, step_logits

    # Raw last-token logits of a single forward (left padding => index -1 is the
    # real last token). Never -inf, unlike generate()'s processed scores.
    out = model(inputs_embeds=in_embeds, attention_mask=attention_mask)
    last_logits = out.logits[:, -1, :].float()
    step_logits = torch.stack(
        [last_logits[:, no_token_id], last_logits[:, yes_token_id]],
        dim=-1
    )
    pred_ids = last_logits.argmax(dim=-1, keepdim=True)
    return pred_ids, step_logits


class _GraphGenMixin:
    """Shared inference-time generation for graph-conditioned CausalLMs.

    Inputs are numerical ``inputs_embeds`` (text token embeddings with graph
    node embeddings already spliced in by ``first_model``), so generation is
    backbone-agnostic and works for any HF CausalLM that supports
    ``generate(inputs_embeds=...)``. For Qwen3/3.5 "thinking" is disabled
    upstream via a closed empty <think></think> block in the prompt template.
    """

    @torch.no_grad()
    def g_step(self, in_embeds, attention_mask, **kwargs):
        return graph_g_step(self, in_embeds, attention_mask, **kwargs)


# --- Qwen3 backbone -------------------------------------------------------
# Qwen3ForCausalLM already implements a standard forward returning loss/logits
# with inputs_embeds + labels, so we only graft on g_step for inference.
try:
    from transformers import Qwen3ForCausalLM

    class Qwen3GLM(_GraphGenMixin, Qwen3ForCausalLM):
        pass
except Exception:  # pragma: no cover - older transformers without Qwen3
    Qwen3GLM = None

# --- Qwen3.5 backbone -----------------------------------------------------
# Qwen3.5 ships as a (multimodal) *ForConditionalGeneration* class; used with
# text-only inputs_embeds it behaves as a CausalLM (no pixel inputs passed).
try:
    from transformers import Qwen3_5ForConditionalGeneration

    class Qwen3_5GLM(_GraphGenMixin, Qwen3_5ForConditionalGeneration):
        pass
except Exception:  # pragma: no cover - older transformers without Qwen3.5
    Qwen3_5GLM = None
