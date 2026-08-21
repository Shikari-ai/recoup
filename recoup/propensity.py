"""P(recovery | action, context): the model the policy maximises against.

Why logistic regression and not something bigger
------------------------------------------------
The policy does not need a ranking of actions. It needs *calibrated
probabilities*, because it multiplies them by rupee amounts to get expected
value. A model that ranks perfectly but reports 0.9 where the truth is 0.4 will
confidently chase receivables that were never coming back, and the error
compounds across a batch.

Logistic regression on well-chosen features is naturally well-calibrated,
trains in seconds with no dependencies, and -- the decisive property here --
every coefficient is readable. When the agent debits someone, the audit trail
can name the features that drove the decision and their signed contributions.
A gradient-boosted ensemble would likely score a little better on AUC and would
cost all of that. For a system that has to explain itself to a risk team, that
is a bad trade.

The feature set is the domain knowledge
---------------------------------------
This is where the payments insight actually lives: the salary-cycle interaction
for insufficient-funds declines, issuer-health interactions for retries, and
the recoverability x action-type interactions that encode "nudging someone
whose card expired is pointless, and retrying it is worse".

Leakage discipline: every feature must be computable strictly *before* the
action is taken. ``extract()`` takes only an event, a proposed action, and a
causal health snapshot -- it is never handed an outcome. This is enforced by
construction rather than by care, because care does not survive refactoring.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .domain import (
    COMMS_ACTIONS,
    DEBIT_ACTIONS,
    Action,
    ActionKind,
    Channel,
    Rail,
    Recoverability,
    RiskEvent,
)
from .issuer_health import HealthSnapshot
from .taxonomy import Classification

IST_OFFSET = timedelta(minutes=330)


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def extract(
    event: RiskEvent,
    classification: Classification,
    action: Action,
    health: HealthSnapshot | None,
    now: datetime,
) -> dict[str, float]:
    """Build the feature vector for one (event, action) pair.

    Sparse dict rather than a dense list so that adding a feature never
    silently shifts the meaning of an existing weight in a saved model.
    """
    f: dict[str, float] = {"bias": 1.0}

    # -- amount, scaled so a Rs 100 and a Rs 100,000 receivable are comparable
    rupee = event.amount_paise / 100.0
    f["amt_log"] = math.log1p(rupee) / 12.0

    # -- how hard have we tried already
    f["attempt_no"] = min(event.attempt_no, 5) / 5.0
    f["is_first_attempt"] = 1.0 if event.attempt_no == 0 else 0.0
    # Effort already spent on this receivable. Every channel of recovery decays
    # with repetition, so without these the policy cannot tell its first move
    # from its fifth and will keep paying full price for a fading return.
    f["actions_taken"] = min(event.actions_taken, 6) / 6.0
    f["comms_taken"] = min(event.comms_taken, 4) / 4.0
    f["untouched"] = 1.0 if event.actions_taken == 0 else 0.0

    # -- staleness: intent and balances both decay
    hours_since = max(0.0, (now - event.occurred_at).total_seconds() / 3600.0)
    f["hours_since_log"] = math.log1p(hours_since) / 6.0

    # -- calendar effects, in IST because that is where the payer's salary lands
    local = action.execute_at + IST_OFFSET
    dom, hour = local.day, local.hour
    # Indian salary credit clusters at month start; balances are systematically
    # higher in the first week and thinnest just before it.
    f["salary_window"] = 1.0 if 1 <= dom <= 7 else 0.0
    f["pre_salary_squeeze"] = 1.0 if 26 <= dom <= 31 else 0.0
    f["biz_hours"] = 1.0 if 9 <= hour < 21 else 0.0
    f["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    f["hour_cos"] = math.cos(2 * math.pi * hour / 24)

    # -- payer history
    seen = event.customer.prior_successes + event.customer.prior_failures
    f["cust_success_rate"] = (event.customer.prior_successes + 1) / (seen + 2)  # Laplace
    f["cust_seen_log"] = math.log1p(seen) / 4.0
    f["cust_new"] = 1.0 if seen == 0 else 0.0

    # -- issuer conditions at decision time
    if health is not None:
        f["issuer_score"] = health.score
        f["issuer_degraded"] = 1.0 if health.degraded else 0.0
        f["issuer_relative"] = _clip(health.relative, 0.0, 1.5) / 1.5
    else:
        f["issuer_score"] = 0.85
        f["issuer_degraded"] = 0.0
        f["issuer_relative"] = 1.0 / 1.5

    # -- deadline pressure: value decays to zero at the deadline
    if event.deadline is not None:
        hrs_left = (event.deadline - action.execute_at).total_seconds() / 3600.0
        f["deadline_pressure"] = _clip(1.0 - hrs_left / 336.0)  # 14d horizon
        f["past_deadline"] = 1.0 if hrs_left <= 0 else 0.0
    else:
        f["deadline_pressure"] = 0.0
        f["past_deadline"] = 0.0

    # -- action shape
    is_debit = action.kind in DEBIT_ACTIONS
    is_comms = action.kind in COMMS_ACTIONS
    target_rail = action.rail or event.rail
    f["is_debit"] = 1.0 if is_debit else 0.0
    f["is_comms"] = 1.0 if is_comms else 0.0
    f["rail_switch"] = 1.0 if target_rail != event.rail else 0.0
    f["rail_known_good"] = 1.0 if target_rail in event.customer.known_rails else 0.0

    # -- scheduling delay chosen by the policy, in log-hours
    delay_h = max(0.0, (action.execute_at - now).total_seconds() / 3600.0)
    f["delay_log"] = math.log1p(delay_h) / 6.0

    # -- categorical one-hots
    f[f"fc_{classification.failure_class.value}"] = 1.0
    f[f"rec_{classification.recoverability.value}"] = 1.0
    f[f"act_{action.kind.value}"] = 1.0
    if action.channel is not Channel.NONE:
        f[f"ch_{action.channel.value}"] = 1.0
    f[f"rail_{target_rail.value}"] = 1.0

    # -- interactions: the domain hypotheses, made learnable rather than assumed
    fc = classification.failure_class.value
    rec = classification.recoverability

    # Insufficient funds is a timing problem. Retrying in the salary window is
    # the single highest-leverage move available to a recovery agent in India.
    f["x_insuff_salary"] = f["salary_window"] if fc == "insufficient_funds" else 0.0
    f["x_insuff_squeeze"] = f["pre_salary_squeeze"] if fc == "insufficient_funds" else 0.0
    f["x_insuff_delay"] = f["delay_log"] if fc == "insufficient_funds" else 0.0

    # Retrying into a live outage wastes the attempt; waiting it out does not.
    f["x_retry_degraded"] = f["issuer_degraded"] if is_debit else 0.0
    f["x_retry_health"] = f["issuer_score"] if is_debit else 0.0

    # Comms only move the needle when a human actually has to do something.
    f["x_comms_customer_action"] = (
        1.0 if (is_comms and rec is Recoverability.CUSTOMER_ACTION) else 0.0
    )
    # ...and are close to useless when the instrument itself is dead, unless
    # they are explicitly asking for a new instrument.
    f["x_comms_instrument_dead"] = (
        1.0
        if (
            is_comms
            and rec is Recoverability.INSTRUMENT_CHANGE
            and action.kind is not ActionKind.REQUEST_INSTRUMENT_UPDATE
        )
        else 0.0
    )
    # Switching rails is the whole play for a dead instrument.
    f["x_altrail_instrument"] = (
        1.0 if (action.kind is ActionKind.RETRY_ALT_RAIL and rec is Recoverability.INSTRUMENT_CHANGE)
        else 0.0
    )
    # A silent retry cannot supply an authentication factor.
    f["x_silent_retry_auth"] = 1.0 if (is_debit and fc == "auth_failed") else 0.0

    f["x_engaged_comms"] = f["cust_success_rate"] if is_comms else 0.0
    # Repetition penalties, per action family. A second escalation on the same
    # invoice is worth far less than the first; so is a fourth SMS.
    f["x_escalate_repeat"] = (
        f["actions_taken"] if action.kind is ActionKind.ESCALATE_HUMAN else 0.0
    )
    f["x_comms_repeat"] = f["comms_taken"] if is_comms else 0.0
    f["x_debit_repeat"] = f["attempt_no"] if is_debit else 0.0
    f["x_amt_escalate"] = f["amt_log"] if action.kind is ActionKind.ESCALATE_HUMAN else 0.0
    return f


@dataclass
class LogisticModel:
    """L2-regularised logistic regression trained by minibatch SGD.

    Pure stdlib and fully seeded: the same data and seed produce byte-identical
    weights, which is what makes the reported backtest numbers reproducible by
    anyone who clones the repo.
    """

    weights: dict[str, float] = field(default_factory=dict)
    # Defaults chosen by a grid search on a *separate* tuning scenario
    # (seed 7), never on the seed the README reports. Tuning on the reported
    # test set is the most common way an honest-looking backtest becomes
    # dishonest. See scripts/tune_model.py.
    l2: float = 1e-4
    lr: float = 0.35
    epochs: int = 40
    batch_size: int = 32
    seed: int = 7
    trained_on: int = 0
    #: Base rate of the training set, used as the prior for unseen contexts.
    base_rate: float = 0.0

    # -- inference ---------------------------------------------------------

    def score(self, feats: dict[str, float]) -> float:
        w = self.weights
        return sum(v * w.get(k, 0.0) for k, v in feats.items())

    def predict_proba(self, feats: dict[str, float]) -> float:
        if not self.weights:
            # Untrained model: fall back to the base rate rather than 0.5, so an
            # un-fitted policy behaves conservatively instead of confidently.
            return self.base_rate or 0.15
        return sigmoid(self.score(feats))

    def contributions(self, feats: dict[str, float], top: int = 6) -> list[tuple[str, float]]:
        """Signed per-feature contributions to the logit, largest first.

        This is the explanation surfaced in the audit trail. It is exact, not
        an approximation like SHAP -- the logit *is* the sum of these terms.
        """
        w = self.weights
        terms = [(k, v * w.get(k, 0.0)) for k, v in feats.items() if k != "bias"]
        terms.sort(key=lambda kv: abs(kv[1]), reverse=True)
        return terms[:top]

    # -- training ----------------------------------------------------------

    def fit(self, X: list[dict[str, float]], y: list[int]) -> "LogisticModel":
        if not X:
            raise ValueError("cannot fit on an empty dataset")
        if len(X) != len(y):
            raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")

        rng = random.Random(self.seed)
        # sorted(): float addition is not associative, so gradient accumulation
        # order changes the low bits of every weight. Deterministic order keeps
        # a fixed seed byte-for-byte reproducible.
        keys = sorted({k for f in X for k in f})
        self.weights = {k: 0.0 for k in keys}
        self.base_rate = sum(y) / len(y)
        # Initialise the intercept at the empirical log-odds so training starts
        # calibrated and spends its budget on structure, not on the base rate.
        p0 = min(max(self.base_rate, 1e-4), 1 - 1e-4)
        self.weights["bias"] = math.log(p0 / (1 - p0))

        idx = list(range(len(X)))
        n = len(idx)
        for epoch in range(self.epochs):
            rng.shuffle(idx)
            lr = self.lr / (1.0 + 0.6 * epoch)  # decay: large steps early, fine later
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                grad: dict[str, float] = {}
                for i in batch:
                    f = X[i]
                    err = sigmoid(self.score(f)) - y[i]
                    for k, v in f.items():
                        grad[k] = grad.get(k, 0.0) + err * v
                m = len(batch)
                for k, g in grad.items():
                    reg = self.l2 * self.weights[k] if k != "bias" else 0.0
                    self.weights[k] -= lr * (g / m + reg)
        self.trained_on = len(X)
        return self

    # -- persistence -------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "weights": {k: round(v, 8) for k, v in sorted(self.weights.items())},
                "l2": self.l2,
                "lr": self.lr,
                "epochs": self.epochs,
                "seed": self.seed,
                "trained_on": self.trained_on,
                "base_rate": self.base_rate,
            },
            indent=2,
            sort_keys=True,
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "LogisticModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        m = LogisticModel(
            weights=dict(d["weights"]),
            l2=d.get("l2", 1e-4),
            lr=d.get("lr", 0.12),
            epochs=d.get("epochs", 40),
            seed=d.get("seed", 7),
        )
        m.trained_on = d.get("trained_on", 0)
        m.base_rate = d.get("base_rate", 0.0)
        return m


# ---------------------------------------------------------------------------
# Evaluation. Calibration is reported alongside discrimination because the
# policy needs probabilities it can multiply by money, not just an ordering.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelReport:
    n: int
    base_rate: float
    auc: float
    brier: float
    #: Expected Calibration Error: mean |predicted - actual| across bins.
    ece: float
    log_loss: float
    bins: list[tuple[float, float, int]]  # (mean_pred, actual_rate, count)

    def format(self) -> str:
        lines = [
            f"n={self.n}  base_rate={self.base_rate:.3f}",
            f"AUC       {self.auc:.4f}   (discrimination: can it rank?)",
            f"Brier     {self.brier:.4f}   (lower is better)",
            f"Log-loss  {self.log_loss:.4f}",
            f"ECE       {self.ece:.4f}   (calibration: are the numbers real?)",
            "",
            "  reliability   predicted -> actual   n",
        ]
        for pred, actual, cnt in self.bins:
            if cnt:
                bar = "#" * int(actual * 30)
                lines.append(f"    {pred:>9.3f} -> {actual:>6.3f}  {cnt:>6}  {bar}")
        return "\n".join(lines)


def auc_score(y: list[int], p: list[float]) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-aware."""
    pairs = sorted(zip(p, y))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, yy in pairs if yy == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = sum(r for r, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def evaluate(y: list[int], p: list[float], n_bins: int = 10) -> ModelReport:
    n = len(y)
    if n == 0:
        return ModelReport(0, 0.0, 0.5, 0.0, 0.0, 0.0, [])
    brier = sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / n
    eps = 1e-12
    ll = -sum(
        yi * math.log(max(pi, eps)) + (1 - yi) * math.log(max(1 - pi, eps))
        for pi, yi in zip(p, y)
    ) / n

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for pi, yi in zip(p, y):
        b = min(int(pi * n_bins), n_bins - 1)
        buckets[b].append((pi, yi))
    bins, ece = [], 0.0
    for b in buckets:
        if not b:
            bins.append((0.0, 0.0, 0))
            continue
        mp = sum(x for x, _ in b) / len(b)
        ma = sum(yy for _, yy in b) / len(b)
        bins.append((mp, ma, len(b)))
        ece += (len(b) / n) * abs(mp - ma)

    return ModelReport(
        n=n,
        base_rate=sum(y) / n,
        auc=auc_score(y, p),
        brier=brier,
        ece=ece,
        log_loss=ll,
        bins=bins,
    )
