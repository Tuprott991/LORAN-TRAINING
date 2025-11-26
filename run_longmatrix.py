import argparse, os, sys, shlex, subprocess
from typing import Any, Dict

try:
    import yaml
except Exception:
    print('[FATAL] Please install pyyaml: pip install pyyaml', file=sys.stderr)
    sys.exit(1)

def build_cli_from_cfg(cfg: Dict[str, Any]) -> list:
    args = []
    for k, v in cfg.items():
        if k == "env":  # env xử lý riêng
            continue
        flag = f'--{k}'
        if isinstance(v, bool):
            if v:
                args.append(flag)
        elif isinstance(v, (int, float)):
            args += [flag, str(v)]
        elif isinstance(v, str):
            if v != '':
                args += [flag, v]
        elif isinstance(v, (list, tuple)):
            for item in v:
                args += [flag, str(item)]
        elif v is None:
            continue
        else:
            args += [flag, str(v)]
    return args

REQUIRED_KEYS = ['train_tsv']  # thêm nếu cần: 'tokenizer', ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='Path to YAML config')
    ap.add_argument('--script', type=str, default='train_longmatrix.py', help='Training script path')
    ap.add_argument('--override', type=str, nargs=argparse.REMAINDER,
                    help='Optional extra CLI args to append at the end')
    ap.add_argument('--dry_run', action='store_true', help='Print command and exit')
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f'[FATAL] YAML not found: {args.config}', file=sys.stderr)
        sys.exit(2)

    with open(args.config, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    cfg = data.get('args', data)

    # Preflight warnings
    for k in REQUIRED_KEYS:
        if k not in cfg:
            print(f'[WARN] Missing key in YAML: {k}', file=sys.stderr)

    # Check script
    if not os.path.exists(args.script):
        print(f'[FATAL] Training script not found: {args.script}', file=sys.stderr)
        sys.exit(3)

    cli = [sys.executable, args.script]
    cli += build_cli_from_cfg(cfg)
    if args.override:
        cli += args.override

    print('[run] CWD =', os.getcwd())
    print('[run] CMD =', ' '.join(shlex.quote(x) for x in cli))

    if args.dry_run:
        return

    # Merge env if provided in YAML
    env = os.environ.copy()
    if isinstance(cfg.get('env'), dict):
        for k, v in cfg['env'].items():
            if v is not None:
                env[str(k)] = str(v)

    proc = subprocess.run(cli, env=env)
    if proc.returncode != 0:
        print(f'[ERROR] Training exited with code {proc.returncode}', file=sys.stderr)
    sys.exit(proc.returncode)

if __name__ == '__main__':
    main()