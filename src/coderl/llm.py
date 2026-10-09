"""Code generators: CPU byte-GRU for verification, pretrained causal LM for research."""
import re
import torch

SYSTEM = "Optimize this C++17 program without changing its input/output behavior. Give a brief plan followed by the complete program in one ```cpp code fence."


def prompt_for(source, history=None):
    feedback = ""
    if history:
        last = history[-1]
        feedback = f"\nPrevious attempt feedback: {str(last.get('feedback', {}))[:3000]}\n"
    return SYSTEM + feedback + "\nOriginal program:\n```cpp\n" + source + "\n```\nOptimized program:\n"


def extract_code(text):
    matches = re.findall(r"```(?:cpp|c\+\+)\s*\n(.*?)```", text, flags=re.S)
    if matches:
        return matches[-1].strip()
    # Permit a bare source response; malformed outputs are rejected by the compiler.
    return text.strip()


class ByteLM(torch.nn.Module):
    """Small randomly initialized generator, for local SFT/GRPO integration runs."""
    eos_id = 256

    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.embedding = torch.nn.Embedding(257, hidden)
        self.rnn = torch.nn.GRU(hidden, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, 257)

    def encode(self, text):
        return list(text.encode())

    def encode_prompt(self, text):
        return self.encode(text)

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode(errors="replace")

    def forward(self, ids):
        return self.head(self.rnn(self.embedding(ids))[0])

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=128, temperature=1.0):
        device = next(self.parameters()).device
        ids = torch.tensor([prompt], device=device)
        logits, state = self.rnn(self.embedding(ids))
        current = self.head(logits[:, -1])
        output = []
        for _ in range(max_new_tokens):
            token = int(torch.multinomial(torch.softmax(current[0]/temperature, -1), 1)) if temperature > 0 else int(current[0].argmax())
            output.append(token)
            if token == self.eos_id:
                break
            value, state = self.rnn(self.embedding(torch.tensor([[token]], device=device)), state)
            current = self.head(value[:, -1])
        return output

    def save(self, path):
        torch.save({"kind": "byte_gru", "hidden": self.hidden, "weights": self.state_dict()}, path)

    @classmethod
    def load(cls, path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(saved["hidden"])
        model.load_state_dict(saved["weights"])
        return model


class HFLM(torch.nn.Module):
    def __init__(self, model_name, device="cpu", lora=False, dtype=None, lora_rank=8):
        super().__init__()
        from pathlib import Path
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)
        adapter = Path(model_name) / "adapter_config.json"
        if adapter.exists():
            from peft import AutoPeftModelForCausalLM
            self.model = AutoPeftModelForCausalLM.from_pretrained(model_name, is_trainable=True, dtype=dtype).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
        if lora and not adapter.exists():
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(self.model, LoraConfig(r=lora_rank, lora_alpha=2*lora_rank, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, task_type="CAUSAL_LM"))
        self.eos_id = self.tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("tokenizer must have an EOS token")
        import inspect
        base = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        self._supports_logit_selection = 'logits_to_keep' in inspect.signature(base.forward).parameters

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_prompt(self, text):
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                     tokenize=True, add_generation_prompt=True, return_dict=False)
        return self.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def forward(self, ids, response_tokens=None):
        kwargs = {"logits_to_keep": response_tokens} if response_tokens and self._supports_logit_selection else {}
        logits = self.model(input_ids=ids, use_cache=False, **kwargs).logits
        return logits[:, -response_tokens:] if response_tokens else logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=512, temperature=1.0):
        ids = torch.tensor([prompt], device=next(self.parameters()).device)
        sampling = {"do_sample": True, "temperature": temperature, "top_k": 0, "top_p": 1.0} if temperature > 0 else {"do_sample": False}
        result = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                     **sampling, pad_token_id=self.eos_id, eos_token_id=self.eos_id)
        return result[0, len(prompt):].tolist()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


class LMPlanner:
    def __init__(self, model, max_new_tokens=1024, max_prompt_tokens=8192, temperature=1.0):
        self.model = model.eval()
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.temperature = temperature

    def propose(self, source, history, attempt):
        prompt = self.model.encode_prompt(prompt_for(source, history))
        if len(prompt) > self.max_prompt_tokens:
            raise ValueError("source plus feedback exceeds model context budget; no silent source truncation")
        completion = self.model.generate(prompt, self.max_new_tokens, temperature=self.temperature)
        text = self.model.decode(completion)
        return {"plan": text.split("```")[0][:2000], "action": "llm_generate", "candidate": extract_code(text), "response": text}


class HybridPlanner:
    """Use a pretrained full-program fallback within the SAME graph attempt budget.

    A fixed source-length gate bounds CPU generation cost. It does not inspect
    task ids, human targets, or benchmark outcomes. Long programs still use the
    learned bounded policy. Every generated program goes through the same tools.
    """
    def __init__(self, edit_planner, lm_planner, max_source_chars=300):
        self.edits = edit_planner
        self.lm = lm_planner
        self.max_source_chars = max_source_chars

    def propose(self, source, history, attempt):
        proposal = self.edits.propose(source, history, attempt)
        generated = any(r.get('action') == 'llm_generate' for r in history)
        if len(source) <= self.max_source_chars and not generated and (history or proposal['action'] == 'identity'):
            return self.lm.propose(source, history, attempt)
        return proposal
