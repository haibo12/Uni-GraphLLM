import warnings

from .conversation import conv_templates, SeparatorStyle


IGNORE_TOKEN_ID = -100


def preprocess(data_name, instruction, target, tokenizer, max_length, mode='train', args=None):
    conv = conv_templates["vicuna_v1_1"].copy()
    assert conv.sep_style == SeparatorStyle.TWO

    roles = conv.roles
    tokenizer.padding_side = 'right' if mode == 'train' else 'left'

    # Qwen3/3.5 use their native chat template with thinking disabled; Llama/vicuna
    # keeps the SeparatorStyle.TWO conversation template above.
    use_chat_template = getattr(tokenizer, 'use_chat_template', False)

    # Apply prompt templates
    conversations = []
    chat_prompt = None  # prompt-only string (chat-template path), used for masking
    # num_token
    tokens = " ".join([f"<Node {i}>" for i in range (1, 1 + args.num_token)])
    
    # remove abstract for citation dataset Cora+Pubmed+wikics+arxiv+WN18RR+FB15K237+chempcba+chemhiv
    if data_name in ['chempcba', 'chemhiv']:
        instruction = instruction.replace("<Node 1>", f': {tokens}')
    elif data_name in ['FB15K237']:
        instruction = instruction.replace("<Node 1>", tokens)
        
    elif data_name in ['wikics']:
        instruction = instruction.replace("<Node 1>", tokens)
        if 'Content: ' in instruction and 'Question: ' in instruction:
            prefix = instruction.split('Content: ')[0]
            content = instruction.split('Content: ')[1].split('Question: ')[0]
            question = instruction.split('Question: ')[1]

            candidate = f"{prefix}Content: {content}Question: {question}"
            while len(content) > 1:
                if use_chat_template:
                    _msgs = [{"role": "user", "content": candidate}]
                    _p = tokenizer.apply_chat_template(
                        _msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                    )
                    _probe = _p + (target + (tokenizer.eos_token or '')) if mode == 'train' else _p
                    tok_len = tokenizer(_probe, return_tensors="pt").input_ids.shape[1]
                else:
                    tmp_conv = conv_templates["vicuna_v1_1"].copy()
                    tmp_roles = tmp_conv.roles
                    tmp_conv.append_message(tmp_roles[0], candidate)
                    if mode == 'train':
                        tmp_conv.append_message(tmp_roles[1], target)
                    else:
                        tmp_conv.append_message(tmp_roles[1], None)
                    tok_len = tokenizer(tmp_conv.get_prompt(), return_tensors="pt").input_ids.shape[1]
                if tok_len <= max_length:
                    break
                content = content[:max(1, int(len(content) * 0.8))]
                candidate = f"{prefix}Content: {content}Question: {question}"
            instruction = candidate
    else:
        instruction = instruction.replace("<Node 1>", tokens)
        # instruction = instruction.replace("<Node 1>", 'tokens')
        # instruction = (instruction.split('Abstract: ')[0] + 'Question: ' + instruction.split('Question: ')[1]).replace("<Node 1>", tokens)
    
    if use_chat_template:
        # Native chat template with thinking disabled. The prompt already ends
        # with the closed <think></think> block, so the model answers directly.
        messages = [{"role": "user", "content": instruction}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        if mode == 'train':
            conversations.append(chat_prompt + target + (tokenizer.eos_token or ''))
        else:
            conversations.append(chat_prompt)
    else:
        conv.append_message(roles[0], instruction)
        # conv.append_message(roles[0], "What is the meaning of AI?")
        if mode == 'train':
            conv.append_message(roles[1], target)
        else:
            conv.append_message(roles[1], None)
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        # padding="max_length",
        max_length=max_length,
        truncation=True,
    ).input_ids
    
    if input_ids.shape[1] >= max_length:
        warnings.warn(f"{data_name}: prompt truncated to max_text_length={max_length} tokens")

    if mode != 'train':
        end_token = tokenizer.eos_token or "</s>"
        targets = tokenizer(
            [target+end_token],
            return_tensors="pt",
            # padding="max_length",
            max_length=100,
            truncation=True,
        ).input_ids
    else:
        targets = input_ids.clone()

        # Mask targets. Only compute loss on the assistant outputs.
        sep = conv.sep + conv.roles[1] + ": "

        # Llama's slow (sentencepiece) tokenizer exposes ``.legacy`` and needs
        # the historical offset arithmetic below. Non-Llama backbones (Qwen's
        # BPE tokenizer) have no such attribute, so we fall back to a robust,
        # tokenizer-agnostic prefix mask.
        if hasattr(tokenizer, 'legacy'):
            for conversation, target in zip(conversations, targets):
                total_len = int(target.ne(tokenizer.pad_token_id).sum())

                turns = conversation.split(conv.sep2)
                cur_len = 1
                target[:cur_len] = IGNORE_TOKEN_ID
                for i, turn in enumerate(turns):
                    if turn == "":
                        break
                    turn_len = len(tokenizer(turn).input_ids)

                    parts = turn.split(sep)
                    if len(parts) != 2:
                        break
                    parts[0] += sep
                    # "-2" is hardcoded for the Llama tokenizer to make the offset correct. the first label is not _, but _label
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                    if i != 0 and not tokenizer.legacy:
                        # The legacy and non-legacy modes handle special tokens differently
                        instruction_len -= 1

                    # Ignore the user instructions
                    target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
                    cur_len += turn_len

                    if i != 0 and not tokenizer.legacy:
                        # The legacy and non-legacy modes handle special tokens differently
                        cur_len -= 1

                target[cur_len:] = IGNORE_TOKEN_ID


                if cur_len < max_length:
                    if cur_len != total_len:
                        target[:] = IGNORE_TOKEN_ID
                        print(
                            f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                            f" #turn = {len(turns) - 1}. (ignored)"
                        )
        else:
            # Tokenizer-agnostic masking: keep loss only on the assistant answer
            # by masking the prompt prefix. Longest-common-prefix in token space
            # makes this robust to BPE merges at the prompt/answer boundary.
            for conversation, target in zip(conversations, targets):
                prefix = chat_prompt if chat_prompt is not None else (conversation.split(sep)[0] + sep)
                prefix_ids = tokenizer(prefix).input_ids
                full_ids = target.tolist()
                n = 0
                while n < len(prefix_ids) and n < len(full_ids) and prefix_ids[n] == full_ids[n]:
                    n += 1
                target[:n] = IGNORE_TOKEN_ID

    is_node = (input_ids >= getattr(tokenizer, 'node_token_start', 32000))
    return dict(
        input_ids=input_ids,
        target_ids=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        is_node = is_node,
        text=conversations[0]
    )
