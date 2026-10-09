"""Verify Transformers integration without downloading a large pretrained model."""
import pytest
import torch

transformers=pytest.importorskip('transformers')
tokenizers=pytest.importorskip('tokenizers')


def test_hf_adapter_generation_scoring_and_sft(tmp_path):
    from transformers import GPT2Config,GPT2LMHeadModel,PreTrainedTokenizerFast
    from tokenizers import Tokenizer,models,pre_tokenizers,decoders
    from coderl.llm import HFLM
    from coderl.train_lm import completion_log_probs,sft
    from coderl.schema import Task,TestCase as Case
    torch.manual_seed(42)
    alphabet=sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab={symbol:i for i,symbol in enumerate(alphabet)}
    vocab['<eos>']=256
    tokenizer=Tokenizer(models.BPE(vocab=vocab,merges=[]))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    hf_tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokenizer,eos_token='<eos>',pad_token='<eos>')
    checkpoint=tmp_path/'tiny'
    hf_tokenizer.save_pretrained(checkpoint)
    model=GPT2LMHeadModel(GPT2Config(vocab_size=257,n_positions=2048,n_embd=16,n_layer=1,n_head=1,
                                   bos_token_id=256,eos_token_id=256,pad_token_id=256))
    model.save_pretrained(checkpoint)
    wrapper=HFLM(str(checkpoint)).eval()
    wrapper.tokenizer.chat_template="{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    chat_prompt=wrapper.encode_prompt('Optimize')
    assert 'assistant:' in wrapper.decode(chat_prompt)
    prompt=wrapper.encode('C++ optimization:')
    response=wrapper.generate(prompt,max_new_tokens=4)
    assert 1<=len(response)<=4
    scores=completion_log_probs(wrapper,prompt,response)
    assert len(scores)==len(response) and torch.isfinite(scores).all()
    ids=torch.tensor([prompt+response])
    logits=wrapper(ids[:,:-1])[0].float().log_softmax(-1)
    expected=torch.stack([logits[len(prompt)-1+i,token] for i,token in enumerate(response)])
    assert torch.allclose(scores,expected,atol=1e-6)
    source='#include<iostream>\nint main(){std::cout<<1;}\n'
    task=Task(task_id='fixture',problem_id='fixture',source=source,target=source,
              tests=[Case(name='x',stdin='',stdout='1')],split='train')
    result=sft(wrapper,[task],tmp_path/'sft',steps=1)
    assert result['eligible_real_train_pairs']==1
    reloaded=HFLM(str(tmp_path/'sft'/'model')).eval()
    assert torch.isfinite(completion_log_probs(reloaded,prompt,response)).all()
