"""Download the exact pretrained base used by the included LoRA adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'Qwen/Qwen2.5-Coder-0.5B-Instruct'
REVISION = 'ea3f2471cf1b1f0db85067f1ef93848e38e88c25'
FILES = ['config.json','generation_config.json','merges.txt','tokenizer.json',
         'tokenizer_config.json','vocab.json','model.safetensors','LICENSE','README.md']


def download(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    provenance = Path(__file__).resolve().parents[1]/'data/model_provenance/revision.json'
    expected = json.loads(provenance.read_text()).get('sha256', {}) if provenance.exists() else {}
    for name in FILES:
        path = root/name
        if not path.exists():
            print('Downloading', name, flush=True)
            url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}?download=true'
            temporary = root/(name+'.tmp')
            with urllib.request.urlopen(url, timeout=180) as response, temporary.open('wb') as output:
                while block := response.read(1024*1024):
                    output.write(block)
            temporary.replace(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in expected and hashes[name] != expected[name]:
            raise ValueError(f'{name} differs from the verified pinned model bytes')
    (root/'revision.json').write_text(json.dumps({'repository':REPO,'revision':REVISION,'sha256':hashes},indent=2))
    return root


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='models/qwen-coder')
    p.add_argument('--bind-adapters', nargs='*', default=[], help='Rebind included LoRA adapters to this local base')
    args = p.parse_args()
    root = download(args.output)
    for adapter in args.bind_adapters:
        path = Path(adapter)/'adapter_config.json'
        config = json.loads(path.read_text())
        config['base_model_name_or_path'] = str(root)
        config['revision'] = REVISION
        path.write_text(json.dumps(config,indent=2))
    print(root)
