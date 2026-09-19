from pathlib import Path
import shutil

PATH = Path("src/environment/simulation.py")
if not PATH.exists():
    raise SystemExit("ERROR: run this script from the repository root.")

text = PATH.read_text()

# This migration must start from the Step 1 version.
if "_decision_transition" in text or "def _close_decision_transition" in text:
    raise SystemExit(
        "ERROR: the revised event-interval Step 2 is still present. "
        "Restore the Step 1 version first with: "
        "git restore src/environment/simulation.py"
    )
if "self._pending_transitions" in text or "def _try_learn" in text:
    raise SystemExit(
        "ERROR: the pending-transition Step 2 already appears to be applied."
    )

old = """        self.completed_tasks: List[Task] = []
        self._episode_reward = 0.0
"""
new = """        self.completed_tasks: List[Task] = []
        self._episode_reward = 0.0
        # Each task owns one RL transition. Its successor state is captured
        # when the next task arrives; its reward becomes available when this
        # task completes. Learning occurs after both events have happened.
        self._pending_transitions = {}
"""
assert text.count(old) == 1, "Expected __init__ block not found exactly once."
text = text.replace(old, new, 1)

old = """    def run(self, duration: float = 500.0) -> List[Task]:
        self.completed_tasks = []
        self._episode_reward = 0.0

        rng = np.random.default_rng(self.seed)
"""
new = """    def run(self, duration: float = 500.0) -> List[Task]:
        self.completed_tasks = []
        self._episode_reward = 0.0
        self._pending_transitions = {}

        rng = np.random.default_rng(self.seed)
"""
assert text.count(old) == 1, "Expected run() reset block not found exactly once."
text = text.replace(old, new, 1)

old = """            task = Task(
                task_id=task_id,
                arrival_time=env.now,
                size_bits=float(rng.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX)),
                complexity=float(rng.uniform(TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX)),
            )
            env.process(self._handle_task(env, task, edge, cloud, rng, sc))

    def _handle_task(self, env, task: Task, edge: EdgeServer, cloud: CloudServer, rng, sc):
        state = self._observe(task, edge, sc)
"""
new = """            task = Task(
                task_id=task_id,
                arrival_time=env.now,
                size_bits=float(rng.uniform(TASK_SIZE_MIN, TASK_SIZE_MAX)),
                complexity=float(rng.uniform(TASK_COMPLEXITY_MIN, TASK_COMPLEXITY_MAX)),
            )

            # Decision states occur at task arrivals. The new state is the
            # successor for the immediately preceding task's transition.
            state = self._observe(task, edge, sc)
            self._attach_next_state(task_id, state)

            env.process(self._handle_task(
                env, task, edge, cloud, rng, sc, state
            ))

    def _handle_task(
        self,
        env,
        task: Task,
        edge: EdgeServer,
        cloud: CloudServer,
        rng,
        sc,
        state: tuple,
    ):
"""
assert text.count(old) == 1, "Expected arrival/_handle_task boundary not found exactly once."
text = text.replace(old, new, 1)

old = """            action = self.agent.act(state)
        task.action_taken = action

        if action == ACTION_LOCAL:
"""
new = """            action = self.agent.act(state)
        task.action_taken = action

        if self.agent is not None and hasattr(self.agent, "learn"):
            self._pending_transitions[task.task_id] = {
                "state": state,
                "action": action,
                "reward": None,
                "next_state": None,
            }

        if action == ACTION_LOCAL:
"""
assert text.count(old) == 1, "Expected action-registration block not found exactly once."
text = text.replace(old, new, 1)

old = """        reward = -(self.w_latency * task.latency + self.w_energy * task.energy)
        self._episode_reward += reward

        if self.agent and hasattr(self.agent, 'learn'):
            next_state = self._observe(task, edge, sc)
            self.agent.learn(state, action, reward, next_state)

        self.completed_tasks.append(task)
        yield env.timeout(0)

    def _observe(self, task: Task, edge: EdgeServer, sc: dict) -> tuple:
"""
new = """        reward = -(self.w_latency * task.latency + self.w_energy * task.energy)
        self._episode_reward += reward

        # Reward belongs to this task's action. Its successor state may already
        # have been observed at the next task arrival, or may arrive later.
        transition = self._pending_transitions.get(task.task_id)
        if transition is not None:
            transition["reward"] = reward
            self._try_learn(task.task_id)

        self.completed_tasks.append(task)
        yield env.timeout(0)

    def _attach_next_state(self, task_id: int, next_state: tuple):
        """Attach the successor state captured at the next decision epoch."""
        previous_id = task_id - 1
        transition = self._pending_transitions.get(previous_id)
        if transition is None:
            return

        transition["next_state"] = next_state
        self._try_learn(previous_id)

    def _try_learn(self, task_id: int):
        """Learn exactly once after both reward and successor state exist."""
        if self.agent is None or not hasattr(self.agent, "learn"):
            return

        transition = self._pending_transitions.get(task_id)
        if transition is None:
            return

        if transition["reward"] is None or transition["next_state"] is None:
            return

        self.agent.learn(
            transition["state"],
            transition["action"],
            transition["reward"],
            transition["next_state"],
        )
        del self._pending_transitions[task_id]

    def _observe(self, task: Task, edge: EdgeServer, sc: dict) -> tuple:
"""
assert text.count(old) == 1, "Expected completion-time learning block not found exactly once."
text = text.replace(old, new, 1)

backup = PATH.with_suffix(PATH.suffix + ".step2-backup")
shutil.copy2(PATH, backup)
PATH.write_text(text)

print(f"Updated {PATH}")
print(f"Backup  {backup}")
print("Step 2 applied: pending per-task transitions with next-state at the next decision epoch.")
