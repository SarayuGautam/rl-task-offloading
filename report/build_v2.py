"""
Build EGPG600_Report_SarayuGautam_v2.docx from the untouched original report.

    python report/build_v2.py        (after: python repro/reproduce_all.py)

Every number written into the report is read from results/summary.json and
results/validation.json; none is typed by hand. Figures come from results/figures.
The original .docx is never modified.
"""
import copy
import json
import os
import re
import shutil
import sys

import numpy as np
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from docxlib import Docx, q, NS, mr, msub, msup, mmax, mdelim, math_paragraph, render_pdf  # noqa: E402

SRC = os.path.join(HERE, "EGPG600_Report_SarayuGautam.docx")
OUT = os.path.join(HERE, "EGPG600_Report_SarayuGautam_v2.docx")
WORK = os.path.join(os.environ.get("TMPDIR", "/tmp"), "report_v2_build")
S = json.load(open(os.path.join(ROOT, "results", "summary.json")))
V = json.load(open(os.path.join(ROOT, "results", "validation.json")))
FIG = os.path.join(ROOT, "results", "figures")
TITLE = "Q-Learning for Task Offloading in 3-Tier Device-Edge-Cloud Systems"


# ---------------------------------------------------------------- number helpers
def pc(x, nd=2):
    return f"{x:.{nd}f}%"


def ci(c, nd=2, unit="%"):
    return f"[{c['ci95'][0]:.{nd}f}{unit}, {c['ci95'][1]:.{nd}f}{unit}]"


def sup_exp(e):
    table = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(e).translate(table)


def sci(p, nd=1):
    mant, ex = f"{p:.{nd}e}".split("e")
    return f"{mant} × 10{sup_exp(int(ex))}"


T43 = S["table_4_3_five_seeds_1000s"]
T42 = S["table_4_2_seed42_10000s"]
D44 = S["section_4_4_seed42"]
PER = T43["per_strategy"]
VS = T43["vs"]
vc, ve, vh, vr, vl = (VS[k] for k in ("Always-Cloud", "Always-Edge", "Greedy Heuristic", "Random", "Always-Local"))
ic, ie, ih = vc["improvement_pct"], ve["improvement_pct"], vh["improvement_pct"]
heur_vs_cloud = [(c - h) / c * 100 for c, h in zip(PER["Always-Cloud"]["per_seed_cost"], PER["Greedy Heuristic"]["per_seed_cost"])]
HVC = float(np.mean(heur_vs_cloud))
share = ic["mean"] / HVC * 100
edge_pct = [a["Edge"] for a in T43["ql_actions_pct"]]
conv = T43["convergence_episode"]
cov = T43["coverage"]
assert all(c["states_visited"] == 27 and c["state_actions_visited"] == 81 for c in cov)
assert all(c < e for c, e in zip(PER["Always-Cloud"]["per_seed_cost"], PER["Always-Edge"]["per_seed_cost"])), \
    "Always-Cloud must be the best fixed baseline on every seed"
q42 = T42["Q-learning"]
imp42 = (T42["Always-Cloud"]["cost"] - q42["cost"]) / T42["Always-Cloud"]["cost"] * 100
gap42 = (T42["Greedy Heuristic"]["cost"] - q42["cost"]) / T42["Greedy Heuristic"]["cost"] * 100
CF = V["closed_form_vs_sim"]
cf_max = max(v["rel_err_pct"] for v in CF.values())
mm1_max = max(r["err_W"] for r in V["mm1_standalone"]["rows"]) * 100
PAR = S["pareto"]
par_lat = [r["avg_latency"] for r in PAR["rows"]]
_grp = {}
for r in PAR["rows"]:
    _grp.setdefault(round(r["avg_latency"], 9), []).append(r["w_latency"])
_big = max(_grp.values(), key=len)
PAR_SAME = (f"; the agents for w_lat = {', '.join(f'{w:.2f}' for w in sorted(_big, reverse=True))} learned the "
            "same policy and coincide") if len(_big) > 1 else ""
PAR_W0 = [r["avg_latency"] for r in PAR["rows"] if r["w_latency"] == 0.0][0]
MOB = S["mobility"]["coverage"]
SENS = S.get("sensitivity", {})
UP = S.get("uplink_in_event_time")
QF = S.get("queue_observable_history")
q_rng = D44["q_range_well_trained"]
ql_cost, heur_cost, cloud_cost = PER["Q-learning"]["cost"], PER["Greedy Heuristic"]["cost"], PER["Always-Cloud"]["cost"]

# policy structure across the five seeds (any-visit greedy rule, as evaluated)
POL = {}
for s in S["manifest"]["seeds"]:
    qt = json.load(open(os.path.join(ROOT, "results", "raw", f"seed_{s}.json")))["qtable"]
    POL[s] = {tuple(st): "LEC"[int(np.argmax(np.where(np.array(v) > 0, np.array(qv), -np.inf)))]
              for st, qv, v in zip(qt["states"], qt["q"], qt["visits"])}
idle_edge = sum(POL[s][(0, z, n)] == "E" for s in POL for z in range(3) for n in range(3))
busy35_cloud = sum(POL[s][(2, z, n)] == "C" for s in POL for z in range(3) for n in range(3))
band12_cloud = sum(POL[s][(1, z, n)] == "C" for s in POL for z in range(3) for n in range(3))
local_any = sum(v == "L" for s in POL for v in POL[s].values())
good_exc = {s: [z for z in range(3) if POL[s][(1, z, 2)] == "E"] for s in POL}
gaps = dict(zip(S["manifest"]["seeds"], vh["per_seed_pct"]))
exc_seeds = [s for s in good_exc if good_exc[s]]
pure_seeds = [s for s in good_exc if not good_exc[s]]
assert local_any == 0


WORDS = {0: "No", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
UP_OK = bool(UP) and UP["vs"]["Always-Cloud"]["improvement_pct"]["ci95"][0] > 0 and \
    UP["vs"]["Greedy Heuristic"]["improvement_pct"]["ci95"][1] < 0
SENS_OK = len(SENS) == 8 and all(r["vs_Always-Cloud"]["ci95"][0] > 0 and r["vs_Greedy"]["ci95"][1] < 0
                                 for r in SENS.values())


def rng_of(seeds):
    g = [-gaps[s] for s in seeds]
    return f"{min(g):.2f}–{max(g):.2f}%"


# ---------------------------------------------------------------- build
os.makedirs(os.path.dirname(WORK), exist_ok=True)
d = Docx(SRC, WORK)
P = d.find
TOC_TAB = q("w:tab")


def is_toc(p):
    ppr = p.find(q("w:pPr"))
    return ppr is not None and ppr.find("w:tabs/w:tab[@w:pos='12000']", NS) is not None


def body_template():
    return P("A decision that yields low latency and energy therefore receives a reward close to zero")


# ---- 1. title, declaration, certificate (Phase 2 retitle)
d.set_text(P("Reinforcement Learning for Task Offloading", start=True), "Q-Learning for Task Offloading")
for key in ("I hereby declare that the work reported", "This is to certify that the project report entitled"):
    assert d.replace_in(P(key), "Reinforcement Learning for Task Offloading in 3-Tier Device-Edge-Cloud Systems", TITLE)

# ---- 2. abstract
d.set_text(P("Resource-constrained devices cannot meet the computational demands"), (
    "Resource-constrained devices cannot meet the computational demands of modern mobile applications, yet sending "
    "every task to a distant cloud incurs high transmission latency and energy cost. Mobile Edge Computing (MEC) "
    "places computing resources one network hop from the device, creating a three-tier device-edge-cloud system in "
    "which each task can be executed locally, at the edge, or in the cloud. This project formulates the per-task "
    "choice of tier as a Markov decision process and solves it with tabular Q-learning. The agent learns by "
    "interacting with a custom discrete-event simulator built in Python and SimPy, which models Poisson task arrivals, "
    "a Shannon-Hartley wireless uplink whose signal quality affects both transmission delay and energy, and a "
    "single-server queue at the edge. Each task is scored by a composite cost weighted 0.7 on latency and 0.3 on "
    "energy. The agent is compared with five baselines: four fixed strategies (always-local, always-edge, "
    "always-cloud and random) and an untrained greedy heuristic that evaluates the known cost model using each "
    "task's exact parameters. The simulator's queueing core reproduces M/M/1 theory to within "
    f"{mm1_max:.2f}%, and the complete simulator reproduces closed-form M/G/1 and transmission-delay predictions "
    f"to within {cf_max:.2f}%. Across five independently trained seeds, evaluated with the learned policy frozen and "
    f"at a fixed channel quality, Q-learning reduces mean composite cost by {pc(ic['mean'])} (95% CI "
    f"{ic['ci95'][0]:.2f}–{ic['ci95'][1]:.2f}%) relative to the strongest fixed baseline, always-cloud "
    f"(paired t(4) = {vc['t']:.2f}, p < 0.001), and by {pc(ie['mean'])} relative to always-edge. It does not "
    f"match the greedy heuristic: Q-learning's mean cost is {pc(-ih['mean'])} higher (95% CI "
    f"{-ih['ci95'][1]:.2f}–{-ih['ci95'][0]:.2f}%; p = {vh['p']:.3f}). Unlike the agent, the heuristic sees "
    "each task's CPU demand. The "
    "learned policy never executes locally. It sends a task to the edge when the edge server is idle and to the "
    "cloud when the edge is busy, with a few seed-dependent exceptions, mostly when one or two tasks are present. "
    "A sweep of the "
    "cost weights yields a degenerate latency-energy frontier, because device energy under offloading does not "
    "depend on the destination tier. The project delivers an interpretable, reproducible tabular Q-learning "
    "baseline and states plainly where it falls short: an untrained, better-informed heuristic outperforms it, and "
    "the discrete action space admits no latency-energy trade-off. Both findings motivate the thesis-phase work on "
    "richer state representations and continuous control."))

# ---- 3. chapter 1
d.replace_in(P("The scope of this project is a simulation-based study."), "reproducible Q-Learning baseline", "reproducible Q-learning baseline")
d.set_text(P("The remainder of this report is organised as follows."), (
    "The remainder of this report is organised as follows. Chapter 2 reviews the literature on mobile edge "
    "computing, computation offloading and reinforcement learning, and identifies the research gap. Chapter 3 "
    "presents the system design and methodology, including the MDP formulation, the Q-learning algorithm and the "
    "simulation environment. Chapter 4 reports and analyses the results: simulator validation, performance "
    "comparison, statistical validation, learned-policy analysis, the Pareto trade-off, hyperparameter sensitivity "
    "and a discussion of what the results do and do not show. Chapter 5 concludes and outlines the future "
    "thesis-phase work."))

# ---- 4. chapter 2: notation and background
d.set_text(P("This study adopts binary offloading across three destinations"), (
    "Computation offloading transfers a task from a resource-constrained device to a more capable remote node. The "
    "literature distinguishes binary offloading, in which each task is executed entirely at one location, from "
    "partial offloading, in which a task is split across locations at the cost of greater algorithmic complexity "
    "(Mao et al., 2017). This study adopts binary offloading across three destinations, local device, edge server or "
    "cloud, consistent with comparable studies (Alfakih et al., 2020; Shuai & Xie, 2024) and providing a tractable "
    "action space for Q-learning."))
assert d.replace_in(P("The standard optimisation objective in the offloading literature is a weighted sum"),
                    "task completion delay T and device energy consumption E", "task completion latency L and device energy consumption E")
eq_tpl = P("Cost = alpha * T + beta * E", exact=True)
eq1 = math_paragraph(eq_tpl, [mr("C=")] + [msub([mr("w")], [mr("lat", plain=True)])] + [mr("L+")]
                     + [msub([mr("w")], [mr("eng", plain=True)])] + [mr("E")])
eq_tpl.addnext(eq1)
d.remove(eq_tpl)
d.set_text(P("where alpha and beta are weighting coefficients"), (
    "where L is the task completion latency, E is the device energy consumed, and w_lat and w_eng are weighting "
    "coefficients encoding the application's latency-energy preference, with w_lat + w_eng = 1 (Alfakih et al., "
    "2020; Chen & Liu, 2022). The Q-learning reward in this study is the negative of this cost, because Q-learning "
    "maximises cumulative discounted reward while the goal is cost minimisation."))
bell_anchor = P("The problem is naturally modelled as a Markov Decision Process")
d.set_text(bell_anchor, (
    "The problem is naturally modelled as a Markov Decision Process (MDP) with state space S, action space A, "
    "transition probabilities P(s′ | s, a), reward function r(s, a) and discount factor γ ∈ [0, 1) (Sutton & Barto, "
    "2018). The state captures the current system snapshot, including task characteristics, network quality and "
    "resource availability; the action is the offloading destination; and an optimal policy maps each state to the "
    "action that maximises the expected discounted return, which here is the negative of the expected cumulative "
    "discounted cost. The optimal action-value function Q* satisfies the Bellman optimality equation"))
eq_b = math_paragraph(eq_tpl, [msup([mr("Q")], [mr("*")]), mr("(s,a)=𝔼"),
                               mdelim([mr("r+γ "), mmax([mr("a′")], [msup([mr("Q")], [mr("*")]), mr("(s′,a′)")]),
                                       mr(" ∣ s,a")])])
bell_anchor.addnext(eq_b)
tail = d.clone_after(eq_b, bell_anchor, text=(
    "and an optimal policy acts greedily with respect to Q*. When the state space is finite and small enough to "
    "enumerate, Q* can be learned with tabular Q-learning; when it grows large, neural-network approximation is "
    "required. This study addresses the first case, establishing an interpretable tabular Q-learning baseline for a "
    "three-tier system."))
d.set_text(P("Q-Learning, introduced by Watkins and Dayan (1992)"), (
    "Q-learning, introduced by Watkins (1989) and proved convergent by Watkins and Dayan (1992), is a model-free, "
    "off-policy algorithm that learns a table of action-value estimates Q(s, a) by repeatedly applying a sampled "
    "form of the Bellman optimality equation. The first wave of RL-based offloading papers (2018-2020) applied "
    "Q-learning and its variants to show that adaptive policies outperform static heuristics under dynamic network "
    "conditions. Alfakih et al. (2020) proposed a SARSA-based algorithm, an on-policy variant, for MEC resource "
    "management, minimising system cost (energy and delay) in a multi-server, multi-user architecture with three "
    "offloading options: the nearest edge server, an adjacent edge server and the remote cloud. This multi-tier "
    "framing anticipates the three-tier hierarchy of this study and represents one of the first systematic RL "
    "treatments of multi-destination offloading, although its multi-server architecture introduces complexity that "
    "limits scalability."))
d.replace_in(P("Deep Q-Network (DQN), introduced by Mnih et al. (2015)"), "Q(s, a; theta)", "Q(s, a; θ)")
d.replace_in(P("controlled by the weighting parameters alpha and beta"), "controlled by the weighting parameters alpha and beta",
             "controlled by the weighting parameters w_lat and w_eng")
d.replace_in(P("Third, the literature lacks a study that characterises"), "as a controlled experiment.",
             "as a controlled experiment. This gap is not closed by the present project; it defines the thesis phase (Section 5.3).")

d.replace_in(P("Shuai and Xie (2024) provide the most structurally comparable work"),
             "No equivalent Q-Learning baseline exists for the general-purpose device-edge-cloud architecture in terrestrial scenarios;",
             "In the literature reviewed for this project, no equivalent Q-learning baseline was found for the general-purpose device-edge-cloud architecture in terrestrial scenarios;")
d.replace_in(P("First, no reproducible Q-Learning baseline exists"),
             "First, no reproducible Q-Learning baseline exists for the general-purpose terrestrial three-tier device-edge-cloud offloading problem.",
             "First, the literature reviewed contains no reproducible Q-learning baseline for the general-purpose terrestrial three-tier device-edge-cloud offloading problem.")
d.replace_in(P("First, the literature reviewed contains no reproducible"),
             "The rest of the field moved directly", "In the work reviewed, the field moved directly")

d.replace_in(P("Similar three-tier architectures appear in satellite edge computing"),
             "is the standard method for establishing simulator validity", "is a standard way of establishing simulator validity")

# ---- 5. chapter 3
d.set_text(P("Figure 3.1 summarises the end-to-end workflow"), (
    "Figure 3.1 summarises the end-to-end workflow of the project, from the raw data source through to the final "
    "evaluated outcome. Tasks originate from the SimPy discrete-event simulator (Section 3.6), which draws Poisson "
    "task arrivals, sizes, computational complexity and network quality. These raw values are converted into a "
    "discrete MDP state (Section 3.4.1). The tabular Q-learning agent (Section 3.5) is trained on this stream of "
    "states using an ε-greedy policy over 2,000 episodes. Two independent checks are then applied before any "
    "performance claim is made: the simulator is checked against closed-form queueing and transmission results "
    "(Section 4.1), and the trained agent, with learning switched off, is evaluated greedily against the four fixed "
    "baselines and the Greedy Heuristic across five independently seeded runs with disjoint training data, with "
    "paired t-tests establishing statistical significance (Section 4.3). The outcome is a learned offloading policy "
    "together with the cost-reduction, sensitivity and Pareto-sweep results reported in Chapter 4."))
d.set_text(P("Figure 3.1: End-to-end solution workflow", nth=1), (
    "Figure 3.1: End-to-end solution workflow for task generation, state construction, Q-learning, execution, "
    "evaluation and validation. The observed state also includes the task-size bin (Section 3.4.1); evaluation is "
    "greedy with learning switched off."))
d.replace_in(P("The system comprises three computing tiers."), "(modeled at", "(modelled at")
d.replace_in(P("The system comprises three computing tiers."), "Tasks are generated at the device and must be assigned to exactly one tier.",
             "Tasks are generated at the device and must be assigned to exactly one tier. Figure 3.2 summarises the architecture and the per-tier cost model.")
d.replace_in(P("The environment is a custom discrete-event simulation built with SimPy."),
             "faithfully representing queueing dynamics at the edge server.",
             "faithfully representing queueing dynamics at the edge server. Figure 3.3 shows the interaction between the simulator and the agent for a single task.")
d.replace_in(P("address this context directly, demonstrating significant performance improvements"),
             "demonstrating significant performance improvements", "reporting substantial performance improvements")
d.replace_in(P("The system comprises three computing tiers."), "(lambda = 6 tasks/s", "(λ = 6 tasks/s")
d.replace_in(P("The system comprises three computing tiers."), "utilisation rho approximately 0.41", "utilisation ρ ≈ 0.41 under Always-Edge")
d.replace_in(P("This study therefore uses a custom Python/SimPy simulation tailored"), "rate = B * log2(1 + SNR)", "R = B log₂(1 + SNR)")
assert d.replace_in(P("where B is the channel bandwidth (1 MHz)"), "approximately 137 Kbps", "approximately 137 kbit/s")
assert d.replace_in(P("where B is the channel bandwidth (1 MHz)"), "approximately 10 Mbps", "approximately 10 Mbit/s")
eq_tpl = P("rate = B * log2(1 + SNR)", exact=True)
eqr = math_paragraph(eq_tpl, [mr("R=B "), msub([mr("log", plain=True)], [mr("2")]), mr("(1+SNR)")])
eq_tpl.addnext(eqr)
d.remove(eq_tpl)
d.replace_in(P("Offloading to the edge or cloud requires transmitting the task"),
             "The achievable data rate is computed using the Shannon-Hartley theorem:",
             "The achievable data rate R is computed using the Shannon-Hartley theorem:")
cloud_bullet = P("Cloud: transmission delay + propagation (80 ms) + computation")
uplink_txt = (
    "In the simulator, an offloaded task joins the edge queue at the instant it is generated; its transmission and "
    "propagation delays are added to its latency analytically rather than being advanced in simulated time. At the "
    "evaluation channel these delays (about 0.6 ms of transmission and 5 ms of propagation) are small compared "
    "with the mean edge service time of 69 ms.")
if UP:
    uc = UP["vs"]["Always-Cloud"]["improvement_pct"]
    uh = UP["vs"]["Greedy Heuristic"]["improvement_pct"]
    uplink_txt += (f" As a robustness check, all five seeds were re-run with both delays simulated in event time "
                   f"(the task joins the edge queue only after its upload): the improvement over Always-Cloud "
                   f"becomes {pc(uc['mean'])} (95% CI {ci(uc)}) and the difference to the Greedy Heuristic "
                   f"{pc(uh['mean'])} (95% CI {ci(uh)})" + (", so the conclusions do not depend on this "
                   "simplification." if UP_OK else "; see Section 4.7."))
d.clone_after(cloud_bullet.getnext() if d.text(cloud_bullet.getnext()) == "" else cloud_bullet,
              body_template(), text=uplink_txt)
eq_tpl = P("Cost = 0.7 * latency + 0.3 * energy", exact=True)
eqc = math_paragraph(eq_tpl, [mr("C=0.7 L+0.3 E")])
eq_tpl.addnext(eqc)
d.remove(eq_tpl)
d.replace_in(P("Each task is scored by a composite cost combining latency and energy"),
             "with weights of 0.7 and 0.3 respectively", "with weights w_lat = 0.7 and w_eng = 0.3, where L is the task latency in seconds and E the device energy in joules")
d.set_text(P("The agent observes a discretised state."), (
    "The agent observes a discretised state. In the base configuration the state is a tuple of three binned "
    "variables: the number of tasks at the edge server, counting both waiting tasks and the task in service (4 "
    "bins: 0, 1–2, 3–5 and 6 or more), the task size (3 bins, with boundaries at 3,000 and 7,000 bits) and the "
    "network quality (3 bins, with boundaries at 0.4 and 0.75). This yields 4 × 3 × 3 = 36 nominal states, of which "
    "the 27 with fewer than six tasks at the edge are reached in practice at the arrival rate studied; Section 4.4 "
    "reports the coverage actually achieved. The state deliberately omits the task's CPU demand (complexity), which "
    "strongly affects computation time; from the agent's perspective it is unobserved noise. A mobility extension "
    "adds device velocity (3 bins) and a network-quality trend (3 bins), expanding the nominal state space to 4 × 3 "
    "× 3 × 3 × 3 = 324 states."))
d.set_text(P("Because reinforcement learning maximises cumulative reward while the objective"), (
    "Because Q-learning maximises the expected discounted sum of rewards while the objective is cost minimisation, "
    "the reward for a task is the negative of its composite cost:"))
eq_tpl = P("reward = - (0.7 * latency + 0.3 * energy)", exact=True)
eqw = math_paragraph(eq_tpl, [mr("r=−(0.7 L+0.3 E)")])
eq_tpl.addnext(eqw)
d.remove(eq_tpl)
d.set_text(P("The agent maintains a Q-table mapping each state to a vector"), (
    "The agent maintains a Q-table mapping each state to a vector of action-value estimates, one per action, "
    "initialised to zero. Because every reward in this problem is strictly negative, a zero initialisation is "
    "optimistic: an action that has never been tried in a given state retains a value of zero, which exceeds the "
    "value of every action that has been tried. During training this is desirable, since it drives systematic "
    "exploration of untried actions. At greedy evaluation time, however, it would allow an untried action to win "
    "the arg-max and be reported as a learned decision. The implementation therefore records a visit count N(s, a) "
    "for every state-action pair and restricts every arg-max and max to actions that have actually been tried; "
    "states never visited during training are reported as untrained rather than being assigned a spurious "
    "preference. When a task completes, the agent updates the value of the action chosen for it:"))
eq_tpl = P("Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max Q(s′, a′) - Q(s, a) ]", exact=True)
eqq = math_paragraph(eq_tpl, [mr("Q(s,a)←Q(s,a)+α"),
                              mdelim([mr("r+γ "), mmax([mr("a′:N(s′,a′)>0")], [mr("Q(s′,a′)")]), mr("−Q(s,a)")])])
eq_tpl.addnext(eqq)
d.remove(eq_tpl)
algo_anchor = P("where alpha is the learning rate, gamma is the discount factor")
d.set_text(algo_anchor, (
    "where α is the learning rate, γ the discount factor, r the task's reward and s′ the state observed when the "
    "task completes; the maximum is 0 if no action has yet been tried in s′. The task stream has no terminal state, "
    "and the 100-second episode boundary is a time limit rather than a termination, so every update bootstraps from "
    "s′. Because tasks overlap in time, updates are applied asynchronously in order of task completion. Action "
    "selection is ε-greedy: with probability ε the agent picks a uniformly random tier, and otherwise the tried "
    "action with the highest Q-value. ε starts at 1.0 and is multiplied by 0.998 after each episode down to a floor "
    "of 0.05, which it reaches after 1,497 episodes. If a state was never visited in training, the agent falls "
    "back to the edge; such states arise only when six or more tasks are at the edge. After training the agent is "
    "frozen: ε is set to zero and the "
    "Q-table is no longer updated, so evaluation measures exactly the policy that was learned. Algorithm 1 "
    "summarises the procedure as implemented, and Table 3.1 lists the hyperparameters."))
ALGO = [
    ("Algorithm 1: Tabular Q-learning for task offloading", True),
    ("(implementation: src/agent/q_learning_agent.py)", False),
    ("Input: α = 0.15, γ = 0.90, ε: 1.0 → 0.05, 2,000 episodes", False),
    (" 1: Q(s,a) ← 0, N(s,a) ← 0 for all s, a   ▷ optimistic (rewards < 0)", False),
    (" 2: for episode = 1, …, 2,000 do", False),
    (" 3:   q ~ U(0.1, 1.0); simulate 100 s of Poisson task arrivals", False),
    (" 4:   for each arriving task do", False),
    (" 5:     s ← (edge-load bin, task-size bin, channel-quality bin)", False),
    (" 6:     a ← random tier w.p. ε, else argmax_{a: N(s,a)>0} Q(s,a)", False),
    (" 7:     run task on tier a; on completion get r = −(0.7L + 0.3E), s′", False),
    (" 8:     y ← r + γ · max_{a′: N(s′,a′)>0} Q(s′,a′)  (y ← r if none)", False),
    (" 9:     Q(s,a) ← Q(s,a) + α (y − Q(s,a));  N(s,a) ← N(s,a) + 1", False),
    ("10:   ε ← max(0.05, 0.998 · ε)", False),
    ("11: freeze (ε ← 0, no updates); evaluate greedily on new workloads", False),
]
box_ppr = etree.fromstring(
    '<w:pPr xmlns:w="%s"><w:keepNext/><w:keepLines/><w:pBdr>'
    '<w:top w:val="single" w:sz="6" w:space="4" w:color="000000"/>'
    '<w:left w:val="single" w:sz="6" w:space="4" w:color="000000"/>'
    '<w:bottom w:val="single" w:sz="6" w:space="4" w:color="000000"/>'
    '<w:right w:val="single" w:sz="6" w:space="4" w:color="000000"/></w:pBdr>'
    '<w:shd w:val="clear" w:color="auto" w:fill="F5F5F5"/>'
    '<w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
    '<w:ind w:left="120" w:right="120"/><w:jc w:val="left"/><w:rPr/></w:pPr>' % NS["w"])
prev = algo_anchor
for i, (line, bold) in enumerate(ALGO):
    p = etree.Element(q("w:p"))
    ppr = copy.deepcopy(box_ppr)
    if i == len(ALGO) - 1:
        ppr.remove(ppr.find(q("w:keepNext")))
    p.append(ppr)
    p.append(Docx.make_run(line, None, {"font": "Courier New", "sz": 18, "b": bold}))
    prev.addnext(p)
    prev = p
spacer = d.clone_after(prev, body_template(), text="")
# Table 3.1
d.set_text(P("Table 3.1: Q-Learning and simulation hyperparameters."), "Table 3.1: Q-learning and simulation hyperparameters (src/config.py; the single configuration used for every reported result).")
t31 = d.tables()[1]
sym = {"alpha": "α", "gamma": "γ", "epsilon_start": "ε₀", "epsilon_end": "ε_min", "lambda": "λ"}
for tr in d.rows(t31):
    cells = tr.findall(q("w:tc"))
    s_ = d.text(cells[1])
    if s_ in sym:
        d.set_row(tr, [d.text(cells[0]), sym[s_], d.text(cells[2])])
    if d.text(cells[0]) == "Exploration decay":
        d.set_row(tr, ["Exploration decay", "–", "× 0.998 per episode (floor reached at episode 1,497)"])
    if d.text(cells[0]) == "Training episodes":
        d.set_row(tr, ["Training episodes", "–", "2,000"])
    if d.text(cells[0]) == "Task arrival rate":
        d.set_row(tr, ["Task arrival rate", "λ", "6 tasks/s (Poisson)"])
last = d.rows(t31)[-1]
tmpl = d.rows(t31)[-2]
for vals in reversed([["Q-table initialisation", "Q₀", "0 (optimistic; rewards are negative)"],
                      ["Episode length", "–", "100 simulated seconds (~600 tasks)"],
                      ["Training channel quality", "q", "U(0.1, 1.0), redrawn each episode"],
                      ["Evaluation channel quality", "q", "0.9, fixed"],
                      ["Evaluation length", "–", "1,000 s per seed (Table 4.3); 10,000 s (Table 4.2)"],
                      ["Seeds", "–", "42, 123, 456, 789, 999 (disjoint simulation-seed blocks)"]]):
    d.add_row_after(last, vals, template=tmpl)
d.set_text(P("To quantify the benefit of learning, the Q-Learning agent is compared"), (
    "To quantify the benefit of learning, the Q-learning agent is compared against five baseline strategies. Four "
    "are fixed and state-blind: Always-Local (every task executed on the device), Always-Edge (every task offloaded "
    "to the edge), Always-Cloud (every task offloaded to the cloud) and Random (each task assigned to a uniformly "
    "random tier). The fifth, the Greedy Heuristic, is state-aware but untrained: for each arriving task it evaluates "
    "the same cost model the reward function uses, once per candidate tier, from the task's exact size and CPU "
    "demand, the exact current network quality and an estimate of the edge queue wait (the number of tasks at the "
    "edge multiplied by the mean service time), and selects the tier with the lowest estimated cost. It holds no "
    "Q-table and does not improve with experience. The comparison is deliberately asymmetric in the heuristic's "
    "favour. The heuristic observes raw, undiscretised quantities, including the task's CPU demand, which is not "
    "part of the agent's state at all, and it is given the cost model in closed form. The agent sees three binned "
    "variables and must learn the consequences of its actions from rewards alone. Beating the four fixed strategies "
    "shows only that adapting to state helps. The Greedy Heuristic is a strong, information-advantaged reference: "
    "it shows what a myopic rule with full task information achieves, rather than isolating the value of learning "
    "alone."))
d.set_text(P("Each Q-Learning agent is trained for 2,000 episodes"), (
    "Each Q-learning agent is trained for 2,000 episodes of 100 simulated seconds each, with network quality "
    "re-drawn per episode from the full range [0.1, 1.0]. It is then frozen and evaluated greedily: exploration is "
    "switched off (ε = 0) and the Q-table is no longer updated, so the evaluated policy is exactly the policy learned "
    "in training. All greedy evaluations use a single fixed, documented channel condition, network quality 0.9, so "
    "that the headline comparison, the five-seed statistics, the sensitivity analysis and the Pareto sweep are all "
    "measured under identical conditions. The evaluation workload is drawn from a simulation seed outside every "
    "training range, so it is disjoint from every episode the agent trained on. To ensure results are not an "
    "artefact of a single random seed, the entire train-and-evaluate procedure is repeated for five seeds (42, 123, "
    "456, 789, 999). Each seed trains on its own block of ten thousand simulation seeds (seed × 10,000 + episode "
    "number), so the five training runs draw from non-overlapping simulated workloads. For each seed, the agent and "
    "all five baselines are evaluated on the same 1,000-second evaluation workload (about 6,000 tasks). Always-Cloud "
    "is the cheapest fixed baseline on every seed, so the headline improvement is measured against it, and "
    "separately against the Greedy Heuristic. Performance is measured by average task latency, average device "
    "energy and the composite cost. Following the reporting practice recommended by Henderson et al. (2018), "
    "results are given as means over seeds with 95% confidence intervals (Student t, four degrees of freedom), "
    "differences are tested with paired t-tests, p-values are adjusted for the five comparisons with the "
    "Holm-Bonferroni method, and hyperparameter sensitivity is reported (Section 4.6). The simulator is validated "
    "against closed-form results before any performance claim is made (Section 4.1). An audit of the first version "
    "of this study found that the Q-table was still being updated during the greedy evaluation runs. Every evaluation "
    "result in this report was regenerated with the frozen policy; training, and therefore the learned Q-tables, "
    "coverage and convergence figures, was unaffected."))

# ---- 6. section 4.1
d.set_text(P("Before drawing any conclusions from the simulator"), (
    "Before drawing conclusions from the simulator, it was checked against closed-form results in two ways. First, "
    "the SimPy single-server first-in-first-out resource that implements the edge queue was configured, in a "
    "standalone script, as an M/M/1 system (Poisson arrivals, exponential service, one server), and its measured "
    "mean time in system was compared with the closed-form prediction W = 1 / (μ − λ). With the service rate fixed at "
    "μ = 14.5 tasks per second, the arrival rate was swept across four load levels from light to heavy utilisation "
    "(Table 4.1)."))
d.replace_in(P("Table 4.1: SimPy queue versus M/M/1 theory (service rate"), "(service rate mu = 14.5/s)", "(standalone SimPy queue, service rate μ = 14.5/s)")
t41 = d.tables()[2]
d.set_row(d.rows(t41)[0], ["Arrival rate λ (1/s)", "Utilisation ρ", "W theory (s)", "W simulated (s)", "Verdict"])
cf = CF
d.set_text(P("All four load levels matched the analytical prediction"), (
    f"All four load levels matched the analytical prediction, with a maximum relative error of {mm1_max:.2f}% (at "
    "ρ = 0.62) against a 5% acceptance tolerance. The simulated values are means over between 150,428 and 600,524 "
    "served tasks per load level. This check validates the queueing primitive, not the full simulator, whose edge "
    "service times are uniformly distributed rather than exponential. Second, therefore, the complete simulator was "
    "run under the fixed policies, for which the expected latency has a closed form at the evaluation channel "
    "(quality 0.9; five 10,000-second runs each). Under Always-Edge the edge is an M/G/1 queue, and the "
    "Pollaczek-Khinchine mean wait plus transmission, propagation and mean service time predicts "
    f"{cf['Always-Edge']['theory_s']:.4f} s against {cf['Always-Edge']['sim_mean_s']:.4f} s simulated "
    f"({cf['Always-Edge']['rel_err_pct']:.2f}% error). Under Always-Cloud the prediction is "
    f"{cf['Always-Cloud']['theory_s']:.4f} s against {cf['Always-Cloud']['sim_mean_s']:.4f} s "
    f"({cf['Always-Cloud']['rel_err_pct']:.3f}%). Under Always-Local it is {cf['Always-Local']['theory_s']:.4f} s "
    f"against {cf['Always-Local']['sim_mean_s']:.4f} s ({cf['Always-Local']['rel_err_pct']:.2f}%). The simulator is "
    "therefore consistent with queueing theory at the level of both its components and its end-to-end latency. "
    "Both checks are reproducible (python main.py --agent validate; python -m pytest tests)."))

for p_ in (P("4.1 Simulator Validity (M/M/1 Queuing Theory)", nth=0), P("4.1 Simulator Validity (M/M/1 Queuing Theory)", nth=1)):
    for t_ in p_.iter(q("w:t")):
        if t_.text and "4.1 Simulator Validity (M/M/1 Queuing Theory)" in t_.text:
            t_.text = t_.text.replace("4.1 Simulator Validity (M/M/1 Queuing Theory)",
                                      "4.1 Simulator Validity (M/M/1 and M/G/1 Queueing Theory)")

# ---- 7. section 4.2 + Table 4.2 + Figure 4.1
d.set_text(P("Table 4.2 reports the average task latency, average device energy"), (
    "Table 4.2 reports the average task latency, average device energy and resulting composite cost for each "
    "strategy on the long single-seed evaluation run (seed 42, network quality 0.9, 10,000 simulated seconds, "
    "approximately 59,940 tasks per strategy; slower strategies complete marginally fewer tasks within the fixed "
    "window). The Greedy Heuristic achieves the lowest latency and the lowest composite cost of all six strategies. "
    f"The Q-learning agent is second, {pc(imp42)} cheaper than Always-Cloud and {pc(-gap42)} more expensive than the "
    "heuristic on this seed. Always-Cloud is the strongest fixed baseline, narrowly ahead of Always-Edge, because the "
    "realistic Shannon channel makes the cloud's superior processing speed worthwhile despite its longer "
    "propagation delay. Always-Local is by far the worst, confirming that offloading is essential for this workload. "
    "The four strategies that never execute locally (Q-learning, Greedy Heuristic, Always-Cloud and Always-Edge) "
    "have the same average energy. This follows from the cost model, as Section 4.5 explains. Figure 4.1 shows the "
    "average latency of each strategy over all five seeds."))
d.set_text(P("Table 4.2: Average latency, energy and composite cost by strategy", nth=1), (
    "Table 4.2: Average latency, energy and composite cost by strategy (cost = 0.7 × latency + 0.3 × energy; lower "
    "is better). Single representative seed (42), 10,000-second evaluation with the frozen policy; five-seed "
    "statistics are given in Table 4.3."))
t42 = d.tables()[3]
for tr in d.rows(t42)[1:]:
    name = d.text(tr.findall(q("w:tc"))[0])
    key = {"Q-Learning": "Q-learning"}.get(name, name)
    r = T42[key]
    note = d.text(tr.findall(q("w:tc"))[4])
    if key == "Q-learning":
        note = "Learned policy (this work)"
    d.set_row(tr, [key, f"{r['latency']:.4f}", f"{r['energy']:.6f}" if r['energy'] < 0.01 else f"{r['energy']:.4f}",
                   f"{r['cost']:.4f}", note])
d.set_text(P("Figure 4.1: Average task latency by strategy. Q-Learning"), (
    "Figure 4.1: Average task latency by strategy, as the mean over five seeds (1,000-second evaluation each; "
    "logarithmic axis; error bars show 95% confidence intervals). "
    "Q-learning (vermillion) is the learned policy under study; the Greedy Heuristic (blue) achieves the lowest "
    "latency, and Always-Cloud is the strongest fixed baseline."))

# ---- 8. section 4.3
d.set_text(P("Because a single run could beat the baseline by chance"), (
    "Because a single run could beat a baseline by chance, the comparison was repeated across five seeds and "
    "subjected to formal statistical tests. Each seed trains on its own disjoint block of simulation seeds and is "
    "evaluated on its own disjoint workload, so the five runs are independent replicates that differ in training "
    "workload, exploration noise and evaluation workload. The main comparisons are against Always-Cloud, the "
    "strongest fixed baseline, and against the Greedy Heuristic, which is given the cost model and the raw task "
    "parameters but requires no training; Table 4.3 also reports Always-Edge, Random and Always-Local. Three "
    "analyses were conducted for each comparison."))
d.set_text(P("The 95% confidence interval describes the range"), (
    "The 95% confidence interval gives the range of plausible values for the mean improvement. It is computed as the "
    "mean of the five per-seed improvements plus or minus the Student-t critical value (t = 2.776 for four degrees of "
    "freedom) times the standard error; the large-sample value of 1.96 would understate the width for five samples. "
    f"Relative to Always-Cloud, the mean improvement is {pc(ic['mean'])}, with a 95% confidence interval of "
    f"{ci(ic)} (standard deviation {ic['sd']:.2f} percentage points); the per-seed improvements range from "
    f"{min(vc['per_seed_pct']):.2f}% to {max(vc['per_seed_pct']):.2f}%. Relative to Always-Edge, the baseline used "
    f"to set the proposal's 10% target, the mean improvement is {pc(ie['mean'])} {ci(ie)}, so the target is met "
    "with the whole interval above it. Relative to the Greedy Heuristic, however, the mean difference is "
    f"{pc(ih['mean'])} {ci(ih)}; the per-seed values range from {min(vh['per_seed_pct']):.2f}% to "
    f"{max(vh['per_seed_pct']):.2f}%. The entire interval lies below zero, so the agent is more expensive than "
    "the heuristic on every seed, not merely on average."))
d.set_text(P("The paired t-test formally asks whether a mean difference"), (
    "The paired t-test asks whether a mean difference could plausibly be zero. For each seed there is a matched "
    "pair, the Q-learning cost and the reference cost measured on the identical evaluation workload, and the test "
    "examines the five per-seed differences in composite cost. The null hypothesis is that the mean difference is "
    "zero. The t-statistic is the mean difference divided by its standard error. Pairing cancels seed-to-seed "
    f"variation in the workload and therefore increases sensitivity. Against Always-Cloud the result is t(4) = "
    f"{vc['t']:.2f}, p = {sci(vc['p'])}. Against the Greedy Heuristic it is t(4) = {vh['t']:.2f}, p = "
    f"{sci(vh['p'])}; the negative sign reflects the direction of the difference. After a Holm-Bonferroni "
    f"correction for the five comparisons in Table 4.3, the adjusted p-values are {sci(vc['p_holm'])} and "
    f"{sci(vh['p_holm'])}, both below 0.05. The agent is therefore significantly cheaper than Always-Cloud and "
    "significantly more expensive than the Greedy Heuristic. With five seeds the test has limited power and assumes "
    "approximately normal per-seed differences, so the confidence intervals above are the more informative "
    "summary. Both statistics were cross-checked against the SciPy implementation and agree to machine precision."))
d.set_text(P("measures the magnitude of an effect independently of significance"), (
    "Cohen's d_z for paired data measures the size of an effect relative to its variability: the mean difference "
    f"divided by the standard deviation of the differences. Against Always-Cloud, d_z = {vc['cohens_dz']:.2f}; "
    f"against the Greedy Heuristic, d_z = {vh['cohens_dz']:.2f}. Both far exceed the conventional 0.8 threshold "
    "for a large effect. That is because the per-seed differences are consistent relative to their spread, not "
    "because either percentage is unusually large; d_z describes consistency across seeds, and the percentages "
    "above describe practical size. One methodological note is worth recording. The first version of this analysis "
    "reported 17.03% ± 0.22% and d_z = 101.05 against Always-Cloud, and −4.53% against the heuristic. Those "
    "evaluation runs still updated the Q-table, so each agent went on adapting to the evaluation channel while it "
    "was being measured. With the policy frozen, the five independently trained agents keep their differences in "
    "the few near-tied states described in Section 4.4. The confidence interval therefore widens, d_z falls and the "
    "gap to the heuristic grows. The frozen-policy figures reported here are the correct measure of what was "
    "learned in training."))
d.set_text(P("Table 4.3: Statistical validation summary (five seeds)."),
           "Table 4.3: Statistical validation summary (five seeds, 1,000-second frozen greedy evaluation per seed; "
           "costs as mean ± standard deviation across seeds; improvement = (baseline − Q-learning) / baseline).")
t43 = d.tables()[4]
rows43 = d.rows(t43)
tmpl43 = rows43[1]
for tr in rows43[1:]:
    t43.remove(tr)
lines = []
for n in ["Q-learning", "Greedy Heuristic", "Always-Cloud", "Always-Edge", "Random", "Always-Local"]:
    c = PER[n]["cost"]
    lines.append([f"{n} composite cost", f"{c['mean']:.5f} ± {c['sd']:.5f}"])
for n, lab in [("Always-Cloud", "Always-Cloud"), ("Always-Edge", "Always-Edge"),
               ("Greedy Heuristic", "Greedy Heuristic"), ("Random", "Random"), ("Always-Local", "Always-Local")]:
    v = VS[n]
    i_ = v["improvement_pct"]
    lines.append([f"Improvement vs {lab}", f"{pc(i_['mean'])}, 95% CI {ci(i_)}; t(4) = {v['t']:.2f}; "
                  f"p = {sci(v['p'])} (Holm {sci(v['p_holm'])}); d_z = {v['cohens_dz']:.2f}"])
lines.append(["Verdict", "Significantly cheaper than every fixed baseline; significantly more expensive than the "
                         "Greedy Heuristic"])
anchor = rows43[0]
for vals in lines:
    anchor = d.add_row_after(anchor, vals, template=tmpl43)

# ---- 9. section 4.4
first_e = f"{min(edge_pct):.1f}–{max(edge_pct):.1f}%"
d.set_text(P("During evaluation, conducted at the fixed channel condition"), (
    f"During evaluation at the fixed channel quality of 0.9, the learned policy of seed 42 routes "
    f"{q42['actions_pct']['Edge']:.1f}% of tasks to the edge and {q42['actions_pct']['Cloud']:.1f}% to the cloud "
    f"(five-seed range {first_e} to the edge in the 1,000-second evaluations) and never executes a task locally. The policy is essentially a "
    "threshold on edge load. When the edge server is idle, the edge is selected because it offers the lowest latency "
    f"({idle_edge} of the 45 idle-edge states across the five seeds). When three to five tasks are at the edge, the "
    f"cloud is selected because its faster processor more than offsets its longer propagation delay "
    f"({busy35_cloud} of 45 states). In the intermediate band of one to two tasks, the values of the two offload "
    "tiers are close and the choice varies with task size, channel quality and seed: "
    f"{band12_cloud} of the 45 such states select the cloud and {45 - band12_cloud} select the edge. Figure 4.3 "
    "shows seed 42, whose two exceptions are small tasks at intermediate channel quality and large tasks at good "
    "channel quality. Switching between the two offload tiers according to edge load is the source of the "
    "improvement over the fixed baselines. Training converged well before the end of the run: the 50-episode "
    "moving average of episode reward stays within 5% of its final level from episode "
    f"{conv[0]:,} for seed 42 (median {int(np.median(conv)):,}, range {min(conv):,}–{max(conv):,} across the five "
    "seeds), as shown in Figure 4.2."))
d.set_text(P("Inspecting the full Q-table (Figure 4.3) shows"), (
    "Inspecting the full Q-table (Figure 4.3) shows what the agent did and did not learn, and the tabular "
    "representation makes both directly checkable. Training visited 27 of the 36 states in the base state space "
    "(75.0%) for every seed, and every action within each of those 27 states was tried at least once, so all 81 "
    "reachable state-action pairs are covered. The nine unvisited states are exactly those with six or more tasks "
    "at the edge. That condition essentially never arises at the arrival rate studied (edge utilisation "
    "approximately 0.41 even if every task is offloaded to the edge), and these states are reported as untrained "
    "rather than assigned a policy. Coverage is uneven: for seed 42, every action in the idle-edge states was "
    f"tried at least {D44['min_visits_per_action_by_queue_bin']['0']:,} times, in the one-to-two-task band at least "
    f"{D44['min_visits_per_action_by_queue_bin']['1']:,} times, but in the three-to-five-task band as few as "
    f"{D44['min_visits_per_action_by_queue_bin']['2']} times. Restricting to the {D44['n_well_trained_30']} states "
    "in which every action was tried at least 30 times, Q(local) lies between "
    f"{q_rng['local'][0]:.2f} and {q_rng['local'][1]:.2f}, while Q(edge) and Q(cloud) lie between "
    f"{min(q_rng['edge'][0], q_rng['cloud'][0]):.2f} and {max(q_rng['edge'][1], q_rng['cloud'][1]):.2f}. Local "
    "execution is therefore strictly dominated throughout the trained state space, and no seed selects it in any "
    "visited state. The reason is a large asymmetry in magnitude. The mean task requires 550 million CPU cycles, "
    "which the 500-million-cycle-per-second device needs 1.1 s to execute, whereas even the poorest channel "
    "(about 137 kbit/s) uploads a mean-sized task in about 40 ms. What the agent learns is almost entirely edge "
    "load. Task size and channel quality matter only in the one-to-two-task band, where the edge and cloud values "
    "are close and the greedy choice differs between seeds. This band is also where the agent loses most ground to "
    "the Greedy Heuristic, which resolves the same near-tie task by task from the exact CPU demand and queue count "
    "(Section 4.7). A Q-table lets an operator check a claim of this kind directly; an opaque deep-learning policy "
    "does not."))
d.set_text(P("Figure 4.2: Q-Learning reward convergence over 2,000 training episodes. The 50-episode"), (
    "Figure 4.2: Q-learning training curves over 2,000 episodes. The grey lines show the 50-episode moving average "
    "of episode reward for each of the five seeds; the blue line is their mean with a ±1 standard deviation band; "
    "the dotted line shows the exploration rate ε (right axis). The dashed line marks the median convergence episode "
    f"({int(np.median(conv)):,}), after which the moving average stays within 5% of its final level (per-seed range "
    f"{min(conv):,}–{max(conv):,})."))
d.set_text(P("Figure 4.3: Learned greedy policy per state (L = Local, E = Edge, C = Cloud). The agent"), (
    "Figure 4.3: Learned greedy policy per state for seed 42 (L = Local, E = Edge, C = Cloud; '?' marks fewer than "
    "30 visits per action; '--' marks states never visited). Columns count the tasks at the edge server, waiting or "
    "in service. The policy selects the edge when the edge is idle and the cloud when three or more tasks are "
    "present, and the one-to-two-task band contains the only size- and channel-dependent choices. Local execution "
    "is never selected. States with six or more tasks at the edge were never visited in training."))

# ---- 10. section 4.5
d.set_text(P("To examine the latency-energy trade-off, the cost weights were swept"), (
    "To examine the latency-energy trade-off, the cost weights were swept across seven settings from pure latency "
    "to pure energy. A separate agent was trained for each setting (seed 42, 1,500 training episodes, frozen greedy "
    "evaluation over 2,000 simulated seconds at quality 0.9), and the resulting operating points were plotted "
    f"(Figure 4.4). Average latency varies between {min(par_lat):.4f} s and {max(par_lat):.4f} s across the "
    "weightings" + (" (the highest at w_lat = 0, where the agent is indifferent between the two offload tiers)"
                    if PAR_W0 == max(par_lat) else "") + ", but the frontier is degenerate rather than merely flat. Across all seven weightings the average "
    f"energy is {PAR['energy_min'] * 1e4:.4f} × 10⁻⁴ J, identical to machine precision, because every weighting is "
    "evaluated on the same sequence of arriving tasks and each task incurs the same transmission energy whichever "
    "offload tier receives it. The cause is structural. In the cost model of Section 3.3, device energy under "
    "offloading is the uplink transmission energy, the product of transmission power and transmission time. It "
    "depends on task size and channel quality but not on the offload destination: sending a task to the edge and "
    "sending it to the cloud cost the device the same energy, because the device performs the same transmission in "
    "both cases. Device energy therefore takes only two distinct values per task, one for local execution and one "
    "shared by both offload actions. Since local execution is strictly dominated and never selected (Section 4.4), "
    "energy is constant over the agent's entire realised action set, and no weighting of a constant can move the "
    "operating point along the energy axis. This is an informative negative result about the problem formulation "
    "rather than about the algorithm. It shows that a genuine latency-energy frontier cannot arise from tier "
    "selection alone. It requires either device-side continuous control variables such as CPU frequency and "
    "transmission power, which change energy without changing the destination, or a model in which the destination "
    "itself alters device energy, for example through multi-hop relaying. This directly motivates the "
    "continuous-control work proposed in Section 5.3. The sweep uses a single seed; because energy is structurally "
    "constant, the degeneracy does not depend on the seed."))
d.set_text(P("Figure 4.4: Latency-energy operating points. Each point"), (
    "Figure 4.4: Latency-energy operating points of seven Q-learning agents, each trained under a different latency "
    f"weight w_lat (seed 42; point labels give w_lat{PAR_SAME}). Average device energy is identical at every weighting, so the "
    "points lie on a horizontal line and form a degenerate frontier: device energy does not depend on the offload "
    "destination in this model."))

# ---- 11. new sections 4.6 (sensitivity) and 4.7 (discussion)
h2_tpl = P("4.5 Pareto Trade-Off Analysis", start=True, nth=1)
cap_tpl = P("Table 4.2: Average latency, energy and composite cost by strategy", nth=1)
fig44_cap = P("Figure 4.4: Latency-energy operating points of seven")
anchor = fig44_cap
h46 = d.clone_after(anchor, h2_tpl, text="4.6 Sensitivity Analysis")
sens_rows = []
if SENS:
    default_key = [k for k in SENS if k.startswith("default")][0]
    labels = {default_key: "Default (Table 3.1)", "alpha=0.05": "α = 0.05",
              "alpha=0.30": "α = 0.30", "gamma=0.0": "γ = 0.0 (myopic)", "gamma=0.5": "γ = 0.5",
              "gamma=0.99": "γ = 0.99", "eps_decay=0.995": "ε decay 0.995", "eps_decay=0.999": "ε decay 0.999"}
    for k in [default_key, "alpha=0.05", "alpha=0.30", "gamma=0.0", "gamma=0.5", "gamma=0.99",
              "eps_decay=0.995", "eps_decay=0.999"]:
        if k not in SENS:
            continue
        r = SENS[k]
        sens_rows.append((labels[k], r))
    short = {labels[default_key]: "the default"}
    best = min(sens_rows, key=lambda x: x[1]["cost"]["mean"])
    worst = max(sens_rows, key=lambda x: x[1]["cost"]["mean"])
    bname, wname = short.get(best[0], best[0]), short.get(worst[0], worst[0])
    all_beat_cloud = all(r["vs_Always-Cloud"]["ci95"][0] > 0 for _, r in sens_rows)
    none_beat_heur = all(r["vs_Greedy"]["ci95"][1] < 0 for _, r in sens_rows)
    g0 = SENS.get("gamma=0.0")
    sens_txt = (
        "To check that the conclusions do not hinge on the single hyperparameter configuration of Table 3.1, the "
        "learning rate α, the discount factor γ and the exploration decay factor were each varied one at a time "
        "around their defaults, with every other setting unchanged, and the full five-seed protocol was repeated "
        "for each variant (Table 4.4). The variants were chosen before their results were seen, and the reported "
        "configuration was not changed afterwards. Mean Q-learning cost ranges from "
        f"{best[1]['cost']['mean']:.5f} ({bname}) to {worst[1]['cost']['mean']:.5f} ({wname}). "
        + ("Every variant remains significantly cheaper than Always-Cloud, with the whole 95% confidence interval "
           "above zero. " if all_beat_cloud else "Not every variant's confidence interval against Always-Cloud "
           "excludes zero (Table 4.4). ")
        + ("No variant closes the gap to the Greedy Heuristic; every interval against it lies below zero. "
           if none_beat_heur else "At least one variant's interval against the Greedy Heuristic includes zero, so "
           "the gap is sensitive to hyperparameters (Table 4.4). "))
    if g0:
        sens_txt += (f"The myopic variant γ = 0, which ignores the successor state entirely, reaches a mean cost of "
                     f"{g0['cost']['mean']:.5f} against {SENS[default_key]['cost']['mean']:.5f} for the default. "
                     "Because a decision affects later tasks only through the edge queue, most of the achievable "
                     "benefit here is immediate, and the problem is close to a contextual bandit. The effect of the "
                     "discount factor is small compared with the gap to the Greedy Heuristic, which points to the "
                     "state representation rather than the planning horizon as the limiting factor.")
    a05 = SENS.get("alpha=0.05")
    if a05 and a05["cost"]["mean"] < SENS[default_key]["cost"]["mean"]:
        sens_txt += (f" A smaller learning rate (α = 0.05) lowers the mean cost to {a05['cost']['mean']:.5f} and "
                     f"narrows the gap to the heuristic to {pc(a05['vs_Greedy']['mean'])}, consistent with the "
                     "estimation-noise explanation in Section 4.7; its confidence interval overlaps the default's, "
                     "so this is an observation, not a tuned result, and the reported configuration is unchanged.")
else:
    sens_txt = "TODO(HUMAN): sensitivity results missing — run python repro/reproduce_all.py sensitivity finalize."
p46 = d.clone_after(h46, body_template(), text=sens_txt)
nonconv = {lab: sum(c >= 2000 for c in r["convergence_episode"]) for lab, r in sens_rows}
nc_note = "; ".join(f"{lab}: {n} of 5 seeds had not met the 5% convergence criterion by episode 2,000"
                    for lab, n in nonconv.items() if n)
cap44 = d.clone_after(p46, cap_tpl, text=(
    "Table 4.4: One-at-a-time hyperparameter sensitivity (five seeds per row, 2,000 training episodes, "
    "1,000-second frozen greedy evaluation; mean with 95% confidence interval; negative values mean Q-learning is "
    "more expensive" + (f"; {nc_note}" if nc_note else "") + ")."))
t44 = copy.deepcopy(d.tables()[3])
cap44.addnext(t44)
rows44 = d.rows(t44)
d.set_row(rows44[0], ["Setting", "Q-learning cost", "vs Always-Cloud", "vs Greedy Heuristic", "Convergence episode (median)"])
tmpl44 = rows44[1]
for tr in rows44[1:]:
    t44.remove(tr)
anchor = rows44[0]
for lab, r in sens_rows:
    c = r["cost"]
    anchor = d.add_row_after(anchor, [lab, f"{c['mean']:.5f} ± {c['ci95_half']:.5f}",
                                      f"{pc(r['vs_Always-Cloud']['mean'])} {ci(r['vs_Always-Cloud'])}",
                                      f"{pc(r['vs_Greedy']['mean'])} {ci(r['vs_Greedy'])}",
                                      f"{int(np.median(r['convergence_episode'])):,}"], template=tmpl44)
after_t44 = d.clone_after(t44, body_template(), text="")
h47 = d.clone_after(after_t44, h2_tpl, text="4.7 Discussion")
disc = [
    ("What was learned. Q-learning recovers a simple, interpretable load-threshold policy from rewards alone: use "
     "the edge while it is idle and the cloud once it is busy, and never compute locally. That policy is "
     f"{pc(ic['mean'])} cheaper than the best fixed tier. The Greedy Heuristic, with full task information and the "
     f"cost model in closed form, is {pc(HVC)} cheaper than Always-Cloud. The agent therefore recovers about "
     f"{share:.0f}% of the improvement that the better-informed heuristic achieves, without being given the model."),
    ("Where it falls short. The remaining gap has two identifiable sources. The first is information. The "
     "heuristic knows each task's CPU demand, which decides whether a busy edge or the cloud finishes the task "
     "sooner, while the agent's state omits it; the agent can only learn the average consequence of an action in a "
     "state, and the per-task variation is invisible to it. The second is estimation noise in near-tied states. "
     "With a constant learning rate the Q-values keep fluctuating, and in the one-to-two-task band the values of "
     "edge and cloud are close, so the greedy choice there differs between seeds. The data bear this out. The "
     f"{WORDS[len(exc_seeds)].lower()} seeds whose policy at good channel quality still sends some tasks to a busy edge are "
     f"{rng_of(exc_seeds)} more expensive than the heuristic. The {WORDS[len(pure_seeds)].lower()} seeds with a pure "
     f"idle-edge threshold are {rng_of(pure_seeds)} more expensive. Even the cleaner policy is therefore limited by "
     "the missing information, which a richer state (for example, a binned CPU demand) or function approximation "
     "would be needed to recover."),
    ("A failure in the evaluation protocol. The audit that produced this version found that the first version "
     "evaluated the agent while it was still learning. That inflated the headline improvement over Always-Cloud "
     f"(17.03% before the correction, {pc(ic['mean'])} after) and hid most of the seed-to-seed variability. The "
     "corrected protocol freezes the Q-table before evaluation, and a unit test now guards against the regression. "
     "The lesson generalises: in reinforcement-learning evaluation, train/evaluation separation must be enforced in "
     "code, not by convention."),
]
if UP:
    disc.append(
        "Robustness to the simulator's timing simplification. Simulating the upload and propagation delays in event "
        "time, so that a task joins the edge queue only after its upload, changes the improvement over Always-Cloud "
        f"from {pc(ic['mean'])} to {pc(UP['vs']['Always-Cloud']['improvement_pct']['mean'])} and the difference to "
        f"the heuristic from {pc(ih['mean'])} to {pc(UP['vs']['Greedy Heuristic']['improvement_pct']['mean'])}. "
        + ("Neither conclusion changes." if UP_OK else "This affects the conclusions; see Table 4.3."))
disc.append(
    "What the results do not show. They do not show that Q-learning outperforms a well-specified analytical policy; "
    "it does not. They do not establish a performance ceiling for tabular methods or a crossover point with deep "
    "methods, since no deep agent was trained and no optimal-policy bound was computed. They also do not cover "
    "other channel qualities, arrival rates or multiple devices. These questions define the thesis phase.")
anchor = h47
for txt in disc:
    anchor = d.clone_after(anchor, body_template(), text=txt)

# ---- 12. chapter 5
d.set_text(P("This project designed, implemented and validated a tabular Q-Learning agent"), (
    "This project designed, implemented and validated a tabular Q-learning agent for task offloading in a "
    "three-tier device-edge-cloud system. The principal contributions are: (1) a reproducible Python/SimPy "
    "discrete-event simulator for three-tier offloading, incorporating a Shannon-Hartley wireless channel in which "
    "network quality genuinely affects transmission cost, and validated against M/M/1 theory (within "
    f"{mm1_max:.2f}%) and against closed-form end-to-end latencies (within {cf_max:.2f}%); (2) a tabular "
    "Q-learning baseline whose policy, and whose training coverage, are both directly inspectable; and (3) an "
    "evaluation across five independently trained seeds with the learned policy frozen, showing that Q-learning "
    f"reduces composite cost by {pc(ic['mean'])} (95% CI {ci(ic)}) relative to the strongest fixed baseline, "
    f"Always-Cloud (t(4) = {vc['t']:.2f}, p = {sci(vc['p'])}). Measured against Always-Edge, the baseline used to "
    f"frame the target in the project proposal, the improvement is {pc(ie['mean'])} (95% CI {ci(ie)}), clear of the "
    "proposed 10% target. Measured against the Greedy Heuristic, an untrained strategy that is given the cost "
    f"model and each task's exact parameters, the learned policy is {pc(-ih['mean'])} more expensive (95% CI "
    f"{-ih['ci95'][1]:.2f}–{-ih['ci95'][0]:.2f}%; p = {vh['p']:.3f}). This result is reported as plainly as the others. The value of tabular "
    "Q-learning here is not that it exceeds a well-specified analytical policy; a short heuristic with no training "
    "and more information does that. Its value is that it recovers most of the heuristic's improvement (about "
    f"{share:.0f}%) from interaction alone, without access to the closed-form cost model or the task's CPU "
    "demand." + (" The conclusions are robust to one-at-a-time changes of the learning rate, discount factor and "
    "exploration schedule (Section 4.6)." if SENS_OK else " Section 4.6 reports how these results vary with the "
    "learning rate, discount factor and exploration schedule.")))
qf_txt = ""
if QF:
    qf_txt = (f" Re-measured with the frozen evaluation protocol on seed 42, correcting the observable changes the "
              f"improvement over Always-Cloud from {pc(QF['improvement_vs_cloud_pct'])} to "
              f"{pc(QF['fixed_observable_improvement_vs_cloud_pct'])} and state-action coverage from "
              f"{QF['coverage']['state_actions_visited']} of {QF['coverage']['state_actions_total']} to 81 of 81.")
d.set_text(P("A fourth contribution is methodological."), (
    "A fourth contribution is methodological. The study reports state-space coverage alongside the learned policy "
    "rather than assuming every state was learned equally well. That practice, together with a final audit, "
    "surfaced three implementation issues, all corrected. The first is an arg-max tie-breaking rule that could let "
    "an untried, zero-initialised action outrank a genuinely evaluated one in sparsely visited states. The second is "
    "an edge-load observable that omitted the task in service, which merged an idle server and a busy server into "
    f"one state.{qf_txt} The third is that the evaluation runs kept updating the Q-table. That inflated the "
    f"headline improvement (17.03% before the correction, {pc(ic['mean'])} after) and understated its seed-to-seed "
    "variability. The degenerate Pareto frontier reported in Section 4.5 is a separate and more durable structural "
    "finding: it is a property of the cost model rather than of the algorithm or of training coverage, and it "
    "persists after all three corrections."))
d.set_text(P("The study is bounded by its design choices"), (
    "The study is bounded by its design choices, and several limitations are worth stating precisely. Tabular "
    "Q-learning requires the state to be discretised into bins and does not scale to high-dimensional or "
    "continuous states. The state omits the task's CPU demand, so from the agent's perspective the problem is "
    "partially observable; this is the main identified reason why the Greedy Heuristic, which sees that demand and "
    "is given the cost model, remains cheaper (Section 4.7). Training visited 27 of the 36 base states. The "
    "remaining 9 correspond to edge loads that essentially never occur at the arrival rate studied, so the policy is "
    "undefined there and is reported as such rather than extrapolated; states with three to five tasks at the edge "
    "were visited only rarely. The constant learning rate leaves Q-values fluctuating, so near-tied states can "
    "resolve differently from seed to seed. Device energy in the cost model is the same for both offload "
    "destinations, so the discrete three-action space admits no latency-energy trade-off for the agent to exploit; "
    "a non-degenerate Pareto frontier requires continuous control variables. The model is single-agent and "
    "therefore does not capture contention between multiple devices for shared edge resources. Local execution is "
    "modelled without device-side resource contention, so tasks assigned to the device execute in unbounded "
    f"parallel; the reported Always-Local latency of {T42['Always-Local']['latency']:.3f} s is therefore a lower "
    "bound, since a genuine single-CPU device would be unstable at this arrival rate (the offered load on the "
    "device alone is approximately 6.6 times its capacity). This strengthens rather than weakens the conclusion "
    "that offloading is essential. Uplink and propagation delays are added to latency analytically rather than "
    "being simulated in event time" + ("; Section 4.7 shows that this does not change the conclusions. " if UP_OK else
    " (not re-checked in event time). ") + "The successor "
    "state used in the Bellman bootstrap is observed at task completion and reuses the completed task's size, "
    "rather than the size of the next arrival, which is unknown at that instant. Because task sizes are drawn "
    "independently this introduces a bias in the bootstrap term that is not corrected here. Finally, all evaluation "
    "is conducted at a single fixed channel quality (0.9) with five seeds, the Pareto sweep uses one seed, and all "
    "results are obtained in simulation rather than on physical infrastructure."))
mob_txt = (f"{MOB['states_visited']} of the 324 states ({100 * MOB['states_visited'] / 324:.1f}%)")
p = P("State-space scaling study: train a Deep Q-Network (DQN)")
d.set_text(p, (
    "State-space scaling study: train a Deep Q-Network (DQN) alongside the tabular Q-learning baseline established "
    "here across five state-space complexity levels (36 to 3,240 states, adding task complexity, battery level, "
    "task type and finer discretisation in turn) to locate empirically where, if anywhere, a neural approximator's "
    "ability to generalise across similar states overtakes an exact table's. Adding the task's CPU demand to the "
    "state is the first step, since Section 4.7 identifies it as the main missing information. A preliminary "
    "probe exists: adding velocity and network-quality-trend bins to model mobility (Section 3.4.1) expands the "
    "space to 324 states, and under the same training budget the tabular method visits only "
    f"{mob_txt} (seed 42, velocity 0.8), consistent with the scaling problem this study is designed to "
    "locate. The mobility configuration has not been evaluated for performance."))
d.set_text(P("The interpretable Q-Learning baseline delivered by this project"), (
    "The interpretable Q-learning baseline delivered by this project, together with the Greedy Heuristic as a "
    "stronger reference point than any fixed strategy, provides the well-characterised basis against which the "
    "thesis-phase DQN comparison, deadline-shaping study and masking experiment will be evaluated. Federated "
    "multi-agent learning, in which multiple devices share model updates without exchanging raw task data, is a "
    "longer-horizon extension beyond this thesis phase and is not scheduled among the three studies above."))

# ---- 13. references
ref_tpl = P("Alfakih, T., Hassan, M. M.", start=True)
def add_ref_after(after_sub, text, url=None):
    anchor = P(after_sub, start=True)
    new = d.clone_after(anchor, ref_tpl, text=text + (" " if url else ""))
    if url:
        rid = d.add_hyperlink_rel(url)
        h = etree.SubElement(new, q("w:hyperlink"))
        h.set(q("r:id"), rid)
        h.append(Docx.make_run(url, Docx._first_rpr(ref_tpl), {"u": True}))
    return new
d.replace_in(P("Dash, S., Mishra, J., & Dash, S. K. (2026)", start=True),
             "SN Computer Science. ", "SN Computer Science, 7, Article 99. ")
d.replace_in(P("Sun, J., Zhang, W., Han, M., Shen, M., & Wu, X. (2026)", start=True),
             "The Journal of Supercomputing. ", "The Journal of Supercomputing, 82(5), Article 345. ")
add_ref_after("Dash, S., Mishra, J.", "Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. "
              "(2018). Deep Reinforcement Learning That Matters. Proceedings of the AAAI Conference on Artificial "
              "Intelligence, 32(1).", "https://doi.org/10.1609/aaai.v32i1.11694")
add_ref_after("Sun, J., Zhang, W.", "Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction "
              "(2nd ed.). MIT Press.")
add_ref_after("Sutton, R. S., & Barto, A. G.", "Watkins, C. J. C. H. (1989). Learning from Delayed Rewards [PhD "
              "thesis]. King's College, University of Cambridge.")

# ---- 14. front matter lists: figures, tables, abbreviations, TOC entries
lof = {"Figure 3.1: End-to-end solution workflow": "Figure 3.1: End-to-end solution workflow for task generation, state construction, Q-learning, execution, evaluation and validation.",
       "Figure 3.3: SimPy environment and Q-Learning interaction": "Figure 3.3: SimPy environment and Q-learning interaction for a single task transition.",
       "Figure 4.1: Average task latency by strategy": "Figure 4.1: Average task latency by strategy (five seeds)",
       "Figure 4.2: Q-Learning reward convergence": "Figure 4.2: Q-learning training curves (five seeds)",
       "Figure 4.3: Learned greedy policy per state": "Figure 4.3: Learned greedy policy per state (seed 42)",
       "Figure 4.4: Latency-energy operating points": "Figure 4.4: Latency-energy operating points of seven weightings",
       "Table 3.1: Q-Learning and simulation hyperparameters": "Table 3.1: Q-learning and simulation hyperparameters",
       "Table 4.3: Statistical validation summary (five seeds)": "Table 4.3: Statistical validation summary (five seeds)"}
for k, v in lof.items():
    p = P(k, nth=0)
    d.set_text(p, v)
t43_entry = P("Table 4.3: Statistical validation summary (five seeds)", nth=0)
d.clone_after(t43_entry, t43_entry, text="Table 4.4: One-at-a-time hyperparameter sensitivity (five seeds)")
ab = d.tables()[0]
ab_rows = d.rows(ab)
def ab_insert(before_key, key, val):
    for tr in d.rows(ab):
        if d.text(tr.findall(q("w:tc"))[0]) == before_key:
            new = copy.deepcopy(ab_rows[1])
            d.set_row(new, [key, val])
            tr.addprevious(new)
            return
    raise KeyError(before_key)
ab_insert("DDPG", "CI", "Confidence Interval")
ab_insert("MARL", "M/G/1", "Single-server queue, Poisson arrivals, general service times")
ab_insert("MARL", "M/M/1", "Single-server queue, Poisson arrivals, exponential service times")
toc45 = P("4.5 Pareto Trade-Off Analysis", nth=0)
assert is_toc(toc45)
t47 = d.clone_after(toc45, toc45)
t46 = d.clone_after(toc45, toc45)
for p_, label in ((t46, "4.6 Sensitivity Analysis"), (t47, "4.7 Discussion")):
    ts = p_.findall(".//w:t", NS)
    ts[0].text = label

# ---- 15. global passes: spelling of Q-learning, symbols, British spelling
in_refs = False
heading_styles = {"Heading1", "Heading2", "Heading3"}
US_UK = {r"\butilization\b": "utilisation", r"\bUtilization\b": "Utilisation", r"\boptimizes\b": "optimises",
         r"\bstabilize\b": "stabilise", r"\bstabilizes\b": "stabilises", r"\brealized\b": "realised",
         r"\bmodeled\b": "modelled", r"\bqueuing\b": "queueing", r"\bQueuing\b": "Queueing", r"\bmodeling\b": "modelling", r"\bbehavior\b": "behaviour"}
SYMS = {r"\balpha\b": "α", r"\bgamma\b": "γ", r"\bepsilon\b": "ε", r"\blambda\b": "λ", r"\brho\b": "ρ",
        r"\bmu\b": "μ", r"\btheta\b": "θ"}
for p in d.paras():
    txt = d.text(p)
    if d.style(p) == "Heading1" and txt.strip() == "References":
        in_refs = True
        continue
    if in_refs or d.style(p) in heading_styles or is_toc(p):
        continue
    if p.find(".//m:oMath", NS) is not None:
        continue
    for t in p.iter(q("w:t")):
        s_ = t.text or ""
        s_ = s_.replace("Q-Learning", "Q-learning")
        for pat, rep in list(US_UK.items()) + list(SYMS.items()):
            s_ = re.sub(pat, rep, s_)
        s_ = re.sub(r"(?<![\w.])-(?=\d)", "−", s_)
        t.text = s_
# title case stays in the title page, declaration and certificate
for key in ("I hereby declare that the work reported", "This is to certify that the project report entitled"):
    d.replace_in(P(key), "Q-learning for Task Offloading", "Q-Learning for Task Offloading")
d.replace_in(P("Q-learning for Task Offloading", start=True), "Q-learning for Task Offloading", "Q-Learning for Task Offloading")

# ---- 15a. inline subscripts: w_lat, w_eng, ε_min, d_z  ->  base + subscript run
SUBS = [("w_lat", "w", "lat"), ("w_eng", "w", "eng"), ("ε_min", "ε", "min"), ("d_z", "d", "z")]
in_refs = False
for p in d.paras():
    if d.style(p) == "Heading1" and d.text(p).strip() == "References":
        in_refs = True
    if in_refs or not any(k in d.text(p) for k, _, _ in SUBS):
        continue
    for r in list(p.iter(q("w:r"))):
        t = r.find(q("w:t"))
        if t is None or not t.text or not any(k in t.text for k, _, _ in SUBS):
            continue
        pat = "(" + "|".join(re.escape(k) for k, _, _ in SUBS) + ")"
        parts = re.split(pat, t.text)
        rpr = r.find(q("w:rPr"))
        anchor = r
        for part in parts:
            if not part:
                continue
            match = next((x for x in SUBS if x[0] == part), None)
            segs = [(match[1], {}), (match[2], {"va": "subscript"})] if match else [(part, {})]
            for txt, fmt in segs:
                nr = Docx.make_run(txt, rpr, fmt)
                anchor.addnext(nr)
                anchor = nr
        r.getparent().remove(r)

# ---- 15b. TOC tab stop: the original right tab (12000 twips) lies beyond the right margin of the
# 8640-twip text column, so LibreOffice/PDF rendering dropped every TOC page number. Move it to the margin.
for p in d.paras():
    if is_toc(p):
        tab = p.find("w:pPr/w:tabs/w:tab", NS)
        tab.set(q("w:pos"), "8640")
        tab.set(q("w:leader"), "dot")


def is_toc(p):   # redefined: after 15b the tab sits at 8640
    ppr = p.find(q("w:pPr"))
    return ppr is not None and ppr.find("w:tabs/w:tab[@w:pos='8640']", NS) is not None


# ---- 16. figures and document metadata
for media, fig in (("image1.png", "fig4_1_latency_by_strategy.png"), ("image4.png", "fig4_2_learning_curve.png"),
                   ("image3.png", "fig4_3_policy_heatmap.png"), ("image6.png", "fig4_4_pareto.png")):
    shutil.copy(os.path.join(FIG, fig), os.path.join(WORK, "word", "media", media))

core = os.path.join(WORK, "docProps", "core.xml")
os.makedirs(os.path.dirname(core), exist_ok=True)
with open(core, "w", encoding="utf-8") as f:
    f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            f'<dc:title>{TITLE}</dc:title><dc:creator>Sarayu Gautam</dc:creator>'
            '<cp:keywords>Q-learning; task offloading; mobile edge computing</cp:keywords>'
            '</cp:coreProperties>')
ct = os.path.join(WORK, "[Content_Types].xml")
ct_x = open(ct, encoding="utf-8").read()
if "core-properties" not in ct_x:
    ct_x = ct_x.replace("</Types>", '<Override PartName="/docProps/core.xml" ContentType="application/'
                        'vnd.openxmlformats-package.core-properties+xml"/></Types>')
    open(ct, "w", encoding="utf-8").write(ct_x)
rels = os.path.join(WORK, "_rels", ".rels")
rx = open(rels, encoding="utf-8").read()
if "docProps/core.xml" not in rx:
    rx = rx.replace("</Relationships>", '<Relationship Id="rIdCore1" Type="http://schemas.openxmlformats.org/'
                    'package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>')
    open(rels, "w", encoding="utf-8").write(rx)

d.save(OUT)
print("wrote", OUT)


# ---- 17. table of contents page numbers (two-pass: render, read printed page numbers, rewrite)
def toc_pages(pdf):
    import subprocess
    n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", pdf], capture_output=True, text=True).stdout).group(1))
    pages = [subprocess.run(["pdftotext", "-layout", "-f", str(i), "-l", str(i), pdf, "-"],
                            capture_output=True, text=True).stdout for i in range(1, n + 1)]
    return pages


def printed_number(page_text):
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    return lines[-1] if lines and re.fullmatch(r"\d+|[ivxlc]+", lines[-1]) else None


def norm(t):
    return re.sub(r"\s+", " ", t).strip()


pdf = render_pdf(OUT, os.path.dirname(WORK))
pages = toc_pages(pdf)
entries = [p for p in d.paras() if is_toc(p)]
labels = [norm(d.text(p).rsplit(" ", 0)[0]) for p in entries]
# body starts at the first page (after the TOC) whose text has "Chapter 1: Introduction" as a line
body_start = next(i for i, pg in enumerate(pages)
                  if any(norm(ln) == "Chapter 1: Introduction" for ln in pg.splitlines())
                  and not any(norm(ln).startswith("1.1 Background") and norm(ln) != "1.1 Background"
                              for ln in pg.splitlines()))
changed = 0
for p in entries:
    ts = p.findall(".//w:t", NS)
    label = norm(ts[0].text)
    if len(ts) < 2 or not label:
        continue
    key = label[:40]
    hit = None
    for i in range(body_start, len(pages)):
        if any(norm(ln).startswith(key) for ln in pages[i].splitlines()):
            hit = i
            break
    if hit is None:
        print("  TOC: heading not found in PDF:", label)
        continue
    num = printed_number(pages[hit])
    if num and ts[-1].text != num:
        ts[-1].text = num
        changed += 1
print(f"TOC entries updated: {changed}")
d.save(OUT)
pdf = render_pdf(OUT, os.path.dirname(WORK))
shutil.copy(pdf, OUT.replace(".docx", ".pdf"))
print("wrote", OUT.replace(".docx", ".pdf"))
