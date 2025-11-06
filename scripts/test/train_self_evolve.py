# scripts/train_self_evolve.py
from metagen_ai.utils.bootstrap import bootstrap
from metagen_ai.datasets.toy_arith import ToyArithDataset
from metagen_ai.training.loop import SelfEvolveTrainer, TrainerConfig
from metagen_ai.roles.builtin import BUILTIN_ROLES
from metagen_ai.roles.generative import generate_and_register_roles

def main():
    cfg = bootstrap("configs/default.yaml")

    # Optionally pre-augment role space once before training
    init_task = {"question": "What is 7 + 9?", "answer": "16"}
    role_lib = dict(BUILTIN_ROLES)
    role_lib, new_roles = generate_and_register_roles(init_task, cfg, role_library=role_lib)
    print("Pre-augmented roles:", [*role_lib.keys()])

    # Plug trainer
    ds = ToyArithDataset(seed=cfg.get("runtime", {}).get("seed", 42), low=1, high=50)
    trainer = SelfEvolveTrainer(cfg=cfg, dataset=ds, role_library=role_lib)

    # Training knobs can also be set in configs/default.yaml under "training"
    tcfg = TrainerConfig(
        steps=int(cfg.get("training", {}).get("steps", 200)),
        prune_every=int(cfg.get("training", {}).get("prune_every", 5)),
        early_exit=bool(cfg.get("controller", {}).get("early_exit", True)),
        log_dir=cfg.get("runtime", {}).get("log_dir", "logs"),
        log_file=cfg.get("training", {}).get("log_file", "train_self_evolve.jsonl"),
        save_every=int(cfg.get("training", {}).get("save_every", 50)),
    )

    summary = trainer.train(tcfg)
    print("== Training Summary ==")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
