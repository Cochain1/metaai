📦 安装
conda create -n metagen python==3.10
conda activate metagen
pip install -r requirements.txt


可选（避免在线下载 SBERT）：

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


并在 configs/default.yaml 指定本地 SBERT 路径（见下）。

🔧 LLM 配置（DeepSeek / 兼容 OpenAI SDK）

在运行代码前在终端中设置 API Key：（下面的命令在终端输入）

export DEEPSEEK_API_KEY="sk-98638328aa79492295d57a125f496064"


编辑 configs/default.yaml：(不用改已经配好了)

llm:
  provider: openai
  base_url: https://api.deepseek.com
  api_key_env: DEEPSEEK_API_KEY
  model: deepseek-chat          # 或 deepseek-reasoner
  temperature: 0.1
  max_tokens: 512


若需使用本地已缓存的 SBERT，指定：

architect:
  g_designer:
    sbert_path: /绝对路径/到/sentence-transformers/all-MiniLM-L6-v2  （这个得改）
 
🚀 运行我们的方法（MetaGen-AI）

主脚本：scripts/run_dataset_eval.py
流程：生成角色 → 自动构图 → 热身 → 一拍裁剪 →（可选）题内/题间自进化 → 评测。

Pure-LLM（推荐，我们的方法）

所有角色均使用 API 大模型（不启用本地算子），支持题内/题间演进。

python scripts/run_dataset_eval.py \
  --config configs/default.yaml \
  --datasets gsm8k_test \
  --mode pure-llm \
  --rounds 2 \
  --seeds 1 \
  --max_examples 200 \
  --sleep_s 0.05 \
  --role_cache_path data/roles/gsm8k_generated_roles_1.jsonl \
  --cache_topk 3 \
  --cache_save_only_correct

Hybrid（LLM + 本地确定性算子）

成本更低、效果依任务而定。（屎）

python scripts/run_dataset_eval.py \
  --config configs/default.yaml \
  --datasets gsm8k_test \
  --mode hybrid \
  --rounds 1 \
  --seeds 1 \
  --max_examples -1

MMLU（多选）（建议在 prompt 中提示“只输出一个选项字母”，你当前的生成角色会自行适配；判分已支持）：

python scripts/run_dataset_eval.py \
  --config configs/default.yaml \
  --datasets mmlu_test \
  --mode pure-llm \
  --rounds 2 \
  --seeds 1 \
  --max_examples -1 \
  --sleep_s 0.05 \
  --role_cache_path data/roles/generated_roles.jsonl \
  --cache_topk 3 \
  --cache_save_only_correct \
  --out logs/metrics/eval_metagen_mmlu.csv


HumanEval（代码生成）（建议把 llm.max_tokens 提到 1024–2048，以容纳代码）：

python scripts/run_dataset_eval.py \
  --config configs/default.yaml \
  --datasets humaneval \
  --mode pure-llm \
  --rounds 2 \
  --seeds 1 \
  --max_examples 10 \
  --sleep_s 0.05 \
  --role_cache_path data/roles/humaneval_generated_roles.jsonl \
  --cache_topk 3 \
  --cache_save_only_correct \
  --dump_traces_dir logs/traces_humaneval \
  --print_failures \
  --cache_debug \
  --out logs/metrics/eval_metagen_humaneval_debug.csv

python scripts/run_dataset_eval.py \
  --config configs/default.yaml \
  --datasets humaneval \
  --mode pure-llm \
  --rounds 2 \
  --seeds 1 \
  --max_examples -1 \
  --sleep_s 0.05 \
  --role_cache_path data/roles/humaneval_generated_roles_1.jsonl \
  --cache_topk 3 \
  --cache_save_only_correct \
  --no_prune \
  --no_feedback \
  --dump_traces_dir logs/traces_humaneval_1 \
  --print_failures \
  --cache_debug \
  --out logs/metrics/eval_metagen_humaneval_guardrails_1.csv

输出位置

指标 CSV：logs/metrics/eval_*.csv

角色缓存（若开启）：data/roles/generated_roles.jsonl

进度条会实时打印 acc / avg_tokens / avg_latency_s。

🧪 单智能体baseline

脚本：scripts/run_baselines.py

Zero-shot CoT

python scripts/run_baselines.py \
  --config configs/default.yaml \
  --dataset gsm8k_test \
  --baseline cot \
  --seeds 1 \
  --max_examples -1 \
  --temperature 0.2 \
  --max_tokens 512 \
  --out logs/metrics/baseline_cot_gsm8k_test.csv

Self-Consistency（k=10）

python scripts/run_baselines.py \
  --config configs/default.yaml \
  --dataset gsm8k_test \
  --baseline selfcons \
  --sc_k 10 \
  --seeds 1 \
  --max_examples -1 \
  --temperature 0.7 \
  --max_tokens 512 \
  --out logs/metrics/baseline_selfcons_k10_gsm8k_test.csv

Tree

1) 多分支多数票
python scripts/run_baselines.py \
  --config configs/default.yaml \
  --dataset gsm8k_test \
  --baseline tree \
  --tree_branching 3 \
  --tree_depth 2 \
  --aggregate majority \
  --temperature 0.7 \
  --max_tokens 512 \
  --seeds 1 \
  --max_examples -1 \
  --out logs/metrics/baseline_tree_b3d2_majority_gsm8k_test.csv

2) 用裁判聚合（judge 读各叶子、给出最终数值）
python scripts/run_baselines.py \
  --config configs/default.yaml \
  --dataset gsm8k_test \
  --baseline tree \
  --tree_branching 3 \
  --tree_depth 2 \
  --aggregate judge \
  --temperature 0.7 \
  --max_tokens 512 \
  --seeds 1 \
  --max_examples -1 \
  --out logs/metrics/baseline_tree_b3d2_judge_gsm8k_test.csv

🗣️ 社区版多代理基线（Debate / STaR）

通用执行器：scripts/run_paper_baseline.py
流程与提示词：位于 configs/paperflows/ 与 configs/prompts/（YAML + 文本）。

Debate（双辩手互评 + 裁判，2 轮）
python scripts/run_paper_baseline.py \
  --config configs/default.yaml \
  --dataset gsm8k_test \
  --flow configs/paperflows/mac_community.yaml \
  --seeds 1 \
  --max_examples -1 \
  --out_csv logs/metrics/paper_mac_community_gsm8k_test.csv

STaR / Teacher–Student（学生 → 老师 → 学生，2 轮）
python scripts/run_paper_baseline.py \
  --config configs/default.yaml \
  --dataset gsm8k_test \
  --flow configs/paperflows/star_community.yaml \
  --seeds 1 \
  --max_examples -1 \
  --out_csv logs/metrics/paper_star_community_gsm8k_test.csv


说明

每个角色的 max_tokens 在对应 YAML 里设置；若需统一为 512，请将包含 judge 在内的所有角色统一到 512。

原始多代理生成会落盘：logs/gens/paper/*.jsonl，便于审计。

🧩 目录结构
configs/
  default.yaml                  # 全局配置（LLM、构图、演进、裁剪）
  paperflows/*.yaml             # 社区/论文风格的多代理流程
  prompts/*/*.txt               # 上述流程的 system/user 提示
data/
  roles/generated_roles.jsonl   # 题间演进的角色缓存（可选）
logs/
  metrics/*.csv                 # 聚合指标
  gens/paper/*.jsonl            # 多代理原始对话记录
scripts/
  run_dataset_eval.py           # 我们的方法主流程
  run_baselines.py              # CoT / Self-Consistency
  run_paper_baseline.py         # 社区版 Debate / STaR
src/metagen_ai/
  architect/g_designer.py       # 自动构图（可选 VGAE 细化）
  role_gen/role_generator.py    # 角色生成与规范化
  graph_ops/runner.py           # DAG 执行与钩子
  feedback/textual_grad.py      # 文本梯度（题内自进化）
  pruning/one_shot.py           # 一拍裁剪
  utils/llm.py                  # OpenAI 兼容客户端（DeepSeek base_url）