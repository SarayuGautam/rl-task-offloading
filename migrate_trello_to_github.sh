#!/bin/bash
set -euo pipefail

# ============================================================
# Trello → GitHub Project Migration Script
# Project: "Graduate Project Progress Tracking" (#2)
# Owner: @me (SarayuGautam)
# ============================================================

PROJECT_NUM=2
OWNER="@me"
PROJECT_ID="PVT_kwHOAk_a-c4BU630"

# --- Field IDs ---
STATUS_FIELD="PVTSSF_lAHOAk_a-c4BU630zhKenec"

# Status option IDs
STATUS_READY="61e4505c"       # To Do → Ready
STATUS_IN_PROGRESS="47fc9ee4" # Doing → In progress
STATUS_DONE="98236657"        # Done → Done

echo "============================================"
echo "  Trello → GitHub Project Migration"
echo "============================================"
echo ""

# --- Step 1: Create "Month" single-select field ---
echo "▸ Creating 'Month' field..."
MONTH_FIELD=$(gh project field-create $PROJECT_NUM --owner "$OWNER" \
  --name "Month" \
  --data-type "SINGLE_SELECT" \
  --single-select-options "Month1,Month2,Month3" \
  --format json | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "  ✓ Month field: $MONTH_FIELD"

# --- Step 2: Create "Category" single-select field ---
echo "▸ Creating 'Category' field..."
CATEGORY_FIELD=$(gh project field-create $PROJECT_NUM --owner "$OWNER" \
  --name "Category" \
  --data-type "SINGLE_SELECT" \
  --single-select-options "Code,Docs,Testing" \
  --format json | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "  ✓ Category field: $CATEGORY_FIELD"

# --- Step 3: Get option IDs for Month and Category ---
echo "▸ Fetching field option IDs..."
FIELDS_JSON=$(gh project field-list $PROJECT_NUM --owner "$OWNER" --format json)

# Parse Month option IDs
MONTH1_OPT=$(echo "$FIELDS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data['fields']:
    if f.get('name') == 'Month':
        for o in f['options']:
            if o['name'] == 'Month1': print(o['id'])
")
MONTH2_OPT=$(echo "$FIELDS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data['fields']:
    if f.get('name') == 'Month':
        for o in f['options']:
            if o['name'] == 'Month2': print(o['id'])
")
MONTH3_OPT=$(echo "$FIELDS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data['fields']:
    if f.get('name') == 'Month':
        for o in f['options']:
            if o['name'] == 'Month3': print(o['id'])
")

# Parse Category option IDs
CODE_OPT=$(echo "$FIELDS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data['fields']:
    if f.get('name') == 'Category':
        for o in f['options']:
            if o['name'] == 'Code': print(o['id'])
")
DOCS_OPT=$(echo "$FIELDS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data['fields']:
    if f.get('name') == 'Category':
        for o in f['options']:
            if o['name'] == 'Docs': print(o['id'])
")
TESTING_OPT=$(echo "$FIELDS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for f in data['fields']:
    if f.get('name') == 'Category':
        for o in f['options']:
            if o['name'] == 'Testing': print(o['id'])
")

echo "  ✓ Month options:    Month1=$MONTH1_OPT  Month2=$MONTH2_OPT  Month3=$MONTH3_OPT"
echo "  ✓ Category options:  Code=$CODE_OPT  Docs=$DOCS_OPT  Testing=$TESTING_OPT"
echo ""

# --- Helper: create item and set fields ---
create_card() {
  local title="$1"
  local body="$2"
  local status_opt="$3"
  local month_opt="$4"
  local category_opt="$5"

  echo "  ▸ Creating: $title"

  # Create draft issue
  ITEM_ID=$(gh project item-create $PROJECT_NUM --owner "$OWNER" \
    --title "$title" \
    --body "$body" \
    --format json | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

  # Set Status
  gh project item-edit \
    --id "$ITEM_ID" --project-id "$PROJECT_ID" \
    --field-id "$STATUS_FIELD" \
    --single-select-option-id "$status_opt" > /dev/null 2>&1

  # Set Month (if provided)
  if [ -n "$month_opt" ]; then
    gh project item-edit \
      --id "$ITEM_ID" --project-id "$PROJECT_ID" \
      --field-id "$MONTH_FIELD" \
      --single-select-option-id "$month_opt" > /dev/null 2>&1
  fi

  # Set Category (if provided)
  if [ -n "$category_opt" ]; then
    gh project item-edit \
      --id "$ITEM_ID" --project-id "$PROJECT_ID" \
      --field-id "$CATEGORY_FIELD" \
      --single-select-option-id "$category_opt" > /dev/null 2>&1
  fi

  echo "    ✓ Done (ID: $ITEM_ID)"
}

# ============================================================
#  DONE LIST — Month1 cards (Trello "Done" → Status "Done")
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Migrating DONE cards..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

create_card \
  "Project proposal submitted" \
  "EGPG 600 proposal — RL for Task Offloading in Edge-Cloud Federated Systems. Submitted Jan 26, 2026." \
  "$STATUS_DONE" "$MONTH1_OPT" "$DOCS_OPT"

create_card \
  "Guided course curriculum designed" \
  "6-module curriculum: MEC Foundations, MDP, Q-Learning, SimPy, Advanced RL, Evaluation. 45 contact hours." \
  "$STATUS_DONE" "$MONTH1_OPT" "$DOCS_OPT"

create_card \
  "Repo structure created" \
  "src/environment, src/agent, src/evaluation, tests, notebooks — full modular layout with __init__.py files." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "config.py — all hyperparameters" \
  "TASK_ARRIVAL_RATE=6.0, EDGE_CPU_SPEED=8000, CLOUD_CPU=30000, α=0.15, γ=0.9, ε_decay=0.998. Single source of truth." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "network_model.py — MEC math" \
  "transmission_delay(), compute_time(), energy formulas. local_cost(), edge_cost(), cloud_cost() implemented. Module 1 formulas." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "edge_server.py — SimPy queue" \
  "simpy.Resource(capacity=1) — auto-queues tasks. process() returns actual queue_wait time. M/M/1 model." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "cloud_server.py — infinite capacity" \
  "No queue — always accepts immediately. 80ms propagation delay. process() yields compute timeout." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "simulation.py — main DES" \
  "Poisson arrivals, _handle_task: observe→act→execute→reward→learn. Network quality randomised per episode." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "base_agent.py — abstract interface" \
  "act(state) and learn(s,a,r,s') abstract methods. Thesis extension: swap QLearningAgent → DQNAgent without changing simulation.py." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "q_learning_agent.py — core RL" \
  "Q-table (defaultdict), epsilon-greedy, Bellman update, end_episode() decay, save()/load() .npz. 18 states learned." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "baselines.py — 4 strategies" \
  "AlwaysLocalAgent, AlwaysEdgeAgent, AlwaysCloudAgent, RandomAgent. All extend BaseAgent." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "dqn_agent.py — thesis scaffold" \
  "Stub with TODO comments: build_network(), _train_step(), replay buffer. Same act()/learn() interface." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "metrics.py — evaluation" \
  "avg_latency(), avg_energy(), composite_cost(), action_distribution(), improvement_over_baseline(), summary()." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "visualizer.py — 3 chart types" \
  "learning_curve(), comparison_bar(), qtable_heatmap(). All save to experiments/results/." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "main.py — CLI runner" \
  "python main.py --agent all runs all baselines + Q-Learning. --episodes, --log_dir flags supported." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "Config tuned — 11.9% improvement" \
  "5 debugging iterations: epsilon decay fix, cloud CPU, network quality variation, arrival rate 6/sec. Target ≥10% achieved." \
  "$STATUS_DONE" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "Learning curve chart generated" \
  "Reward converges from -157 (ep 1) to -47 (ep 2000). 50-episode moving average shown. Saved as learning_curve.png." \
  "$STATUS_DONE" "$MONTH1_OPT" "$DOCS_OPT"

create_card \
  "Comparison bar chart generated" \
  "Q-Learning avg_latency=0.091s vs best baseline 0.104s. All 5 strategies compared. comparison_latency.png saved." \
  "$STATUS_DONE" "$MONTH1_OPT" "$DOCS_OPT"

create_card \
  "Q-table heatmap generated" \
  "3 grids (per network quality). Policy: queue=empty → Edge, queue>0 → Cloud. Logical and expected. qtable_heatmap.png." \
  "$STATUS_DONE" "$MONTH1_OPT" "$DOCS_OPT"

create_card \
  "Git tag v0.1-simulation-env" \
  "Simulation runs 10k tasks, baselines working, Q-Learning beats all baselines by 11.9%. First milestone committed." \
  "$STATUS_DONE" "$MONTH1_OPT" ""

# ============================================================
#  DOING LIST (Trello "Doing" → Status "In progress")
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Migrating DOING cards..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

create_card \
  "GitHub push + v0.1 tag live" \
  "Unzip repo, pip install -r requirements.txt, python main.py --agent all verify karney, GitHub ma push garney." \
  "$STATUS_IN_PROGRESS" "$MONTH1_OPT" "$CODE_OPT"

create_card \
  "Month 1 code bujhney" \
  "config.py → network_model.py → simulation.py → q_learning_agent.py order maa padha. Notes sanga connect gara." \
  "$STATUS_IN_PROGRESS" "$MONTH1_OPT" ""

# ============================================================
#  TO DO LIST (Trello "To Do" → Status "Ready")
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Migrating TO DO cards..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

create_card \
  "statistical_tests.py — 5 seeds" \
  "Run Q-Learning with seeds [42,123,456,789,999]. Compute 95% confidence interval. Prove 11.9% is statistically significant." \
  "$STATUS_READY" "$MONTH2_OPT" "$CODE_OPT"

create_card \
  "Jupyter notebook 01 — MEC math" \
  "Interactive notebook: transmission delay, energy formulas with sliders. Module 1 ko hands-on version." \
  "$STATUS_READY" "$MONTH2_OPT" "$DOCS_OPT"

create_card \
  "Jupyter notebook 02 — Q-Learning debug" \
  "Episode-by-episode reward plot, action histogram, Q-table heatmap interactive. Module 3+5 visual." \
  "$STATUS_READY" "$MONTH2_OPT" "$DOCS_OPT"

create_card \
  "Hyperparameter sensitivity analysis" \
  "Vary α=[0.05,0.1,0.15,0.3], γ=[0.7,0.9,0.95], ε_decay=[0.99,0.998,0.999]. Plot performance vs each param." \
  "$STATUS_READY" "$MONTH2_OPT" "$CODE_OPT"

create_card \
  "Git tag v0.2-qlearning-agent" \
  "After stats validation + notebooks + hyperparameter analysis. Agent convergence proven scientifically." \
  "$STATUS_READY" "$MONTH2_OPT" ""

create_card \
  "DQN implementation — PyTorch" \
  "Complete dqn_agent.py: _build_network() 3-layer MLP, experience replay buffer (capacity=10k), target network, _train_step() batch update." \
  "$STATUS_READY" "$MONTH3_OPT" "$CODE_OPT"

create_card \
  "DQN vs Q-Learning comparison" \
  "Run both on same scenarios. Document: states handled, convergence speed, final performance. Thesis justification." \
  "$STATUS_READY" "$MONTH3_OPT" "$TESTING_OPT"

create_card \
  "Final evaluation report" \
  "Publication-quality results: all metrics, statistical tests, visualizations. Basis for thesis proposal." \
  "$STATUS_READY" "$MONTH3_OPT" "$DOCS_OPT"

create_card \
  "Code documentation complete" \
  "All docstrings, README updated, module-level comments done. Ready for thesis supervisor handover." \
  "$STATUS_READY" "$MONTH3_OPT" "$DOCS_OPT"

create_card \
  "Git tag v1.0-project-complete" \
  "All tests pass, Q-Learning beats baselines ≥10% with CI, DQN scaffold tested, docs complete." \
  "$STATUS_READY" "$MONTH3_OPT" ""

echo ""
echo "============================================"
echo "  ✅ Migration complete!"
echo "  📋 32 cards migrated to project #$PROJECT_NUM"
echo "  🔗 https://github.com/users/SarayuGautam/projects/2"
echo "============================================"
