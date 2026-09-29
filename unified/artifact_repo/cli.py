#!/usr/bin/env python3
"""CLI entry point for the artifact repository.

Usage:
    python -m artifact_repo.cli [command]

Commands:
    gui          Open the desktop model picker (default)
    scan         Scan model directories, import into repository
    list         List all models in the repository
    info  <id>   Show details for a model
    tags         List all tags
    runs         List recent training runs
    stats        Show repository statistics
    launch <id>  Select and launch a model as server
    train <id>   Launch training for a model
"""
import sys, os, json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
from artifact_repo.repository import ArtifactRepository
from artifact_repo.database import DB_PATH


def cmd_gui(repo, args):
    from artifact_repo.picker import ArtifactPicker
    app = ArtifactPicker(repo)
    app.run()


def cmd_scan(repo, args):
    n = repo.scan_and_import()
    print(f"Imported/updated {n} models")
    s = repo.stats()
    print(f"Repository: {s['models']} models, {s['tags']} tags, {s['adapters']} adapters")


def cmd_list(repo, args):
    tag = ''
    search = ''
    sort = 'updated'
    for i, a in enumerate(args):
        if a == '--tag' and i + 1 < len(args): tag = args[i + 1]
        if a == '--search' and i + 1 < len(args): search = args[i + 1]
        if a == '--sort' and i + 1 < len(args): sort = args[i + 1]

    models = repo.list_models(tag=tag, search=search, sort=sort)
    if not models:
        print("No models found. Run 'scan' first.")
        return
    print(f"{'ID':<4} {'Name':<55} {'Size':<7} {'Arch':<14} {'Ctx':<6} {'Rating':<6}")
    print("-" * 100)
    for m in models:
        rating = m.get('rating', 0) or 0
        stars = ('*' * rating + '.' * (5 - rating)) if rating else '-'
        vis = ' [V]' if m.get('mmproj') else ''
        print(f"{m['id']:<4} {(m['filename'] + vis)[:55]:<55} "
              f"{m['size_gb']:>5.1f}G{'':1} {m['architecture'][:14]:<14} "
              f"{str(m['context_length'] or ''):<6} {stars}")
    print(f"\n{len(models)} models")


def cmd_info(repo, args):
    if not args:
        print("Usage: info <model_id>")
        return
    model_id = int(args[0])
    m = repo.get_model(model_id)
    if not m:
        print(f"Model #{model_id} not found")
        return
    for k, v in m.items():
        if k == 'tags':
            print(f"{k}: {v or '(none)'}")
        elif k == 'rating':
            v = int(v or 0)
            print(f"{k}: {'*' * v + '.' * (5 - v) if v else 'unrated'}")
        elif k == 'notes':
            print(f"{k}: {(v or '(empty)')[:100]}")
        else:
            print(f"{k}: {v or '-'}")
    adapters = repo.list_adapters(base_model_id=model_id)
    if adapters:
        print(f"\nAdapters ({len(adapters)}):")
        for a in adapters:
            print(f"  [{a['id']}] {a['name']} ({a['method']}, rank={a['rank']})")
    runs = repo.list_training_runs(base_model_id=model_id)
    if runs:
        print(f"\nTraining runs ({len(runs)}):")
        for r in runs:
            print(f"  [{r['id']}] {r['method']} - {r['status']} ({r['created']})")


def cmd_tags(repo, args):
    tags = repo.list_tags()
    if not tags:
        print("No tags defined")
        return
    for t in tags:
        print(t)


def cmd_runs(repo, args):
    runs = repo.list_training_runs()
    if not runs:
        print("No training runs")
        return
    for r in runs:
        print(f"[{r['id']}] Method={r['method']} Status={r['status']} "
              f"Model={r['base_model_id']} Dataset={r['dataset'][:30] if r['dataset'] else '-'} "
              f"Created={r['created']}")


def cmd_stats(repo, args):
    s = repo.stats()
    print(f"Repository: {DB_PATH}")
    print(f"  Models:        {s['models']}")
    print(f"  Tags:          {s['tags']}")
    print(f"  Adapters:      {s['adapters']}")
    print(f"  Training runs: {s['training_runs']}")
    print(f"  Datasets:      {s['datasets']}")


def cmd_launch(repo, args):
    if not args:
        print("Usage: launch <model_id> [--port PORT]")
        return
    model_id = int(args[0])
    port = 8080
    for i, a in enumerate(args):
        if a == '--port' and i + 1 < len(args): port = int(args[i + 1])
    result = repo.select_and_launch(model_id, port=port)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Launched on port {result.get('port')} (PID {result.get('pid')}): {result.get('model')}")


def cmd_train(repo, args):
    if not args:
        print("Usage: train <model_id> [--method sft|dpo|grpo] [--dataset NAME]")
        return
    model_id = int(args[0])
    method = 'sft'
    dataset = ''
    for i, a in enumerate(args):
        if a == '--method' and i + 1 < len(args): method = args[i + 1]
        if a == '--dataset' and i + 1 < len(args): dataset = args[i + 1]
    result = repo.launch_training(model_id, method=method, dataset=dataset)
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Training run #{result['run_id']} launched ({result['status']})")


def main():
    import sys
    cmds = {
        'gui': cmd_gui, 'scan': cmd_scan, 'list': cmd_list,
        'info': cmd_info, 'tags': cmd_tags, 'runs': cmd_runs,
        'stats': cmd_stats, 'launch': cmd_launch, 'train': cmd_train,
    }
    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        return
    cmd = args[0]
    cmd_args = args[1:]
    repo = ArtifactRepository()
    if cmd in cmds:
        if cmd == 'gui':
            cmd_gui(repo, cmd_args)
        else:
            cmds[cmd](repo, cmd_args)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == '__main__':
    main()
