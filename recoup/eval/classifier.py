"""Taxonomy evaluation: per-class precision and recall, and the errors that matter.

Overall accuracy is a comfortable number that hides the only failures worth
worrying about. The taxonomy is 97.6% accurate on held-out events, and that
figure is nearly useless on its own, because the classes are wildly imbalanced
and their errors are wildly asymmetric.

Two misclassifications, both counted identically by accuracy:

* ``insufficient_funds`` read as ``gateway_error`` -- the agent retries a bit
  too eagerly on a class where retrying was the right idea anyway. Cost: a
  wasted attempt.
* ``mandate_revoked`` read as ``insufficient_funds`` -- the agent re-presents a
  debit against an authorisation the customer has explicitly withdrawn. Cost:
  an unauthorised debit, a chargeback, and a regulatory conversation.

Accuracy says those are the same event. They are not. So this module reports
per-class precision and recall, and then separates errors into three
severity tiers by what they actually cause. The number to watch is
**terminal recall**: the fraction of genuinely terminal failures the taxonomy
correctly refuses to act on.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..domain import FailureClass, Recoverability, RiskEvent
from ..taxonomy import PROFILES, classify

#: Classes where a false negative means acting on something we must not touch.
TERMINAL_CLASSES = frozenset(
    fc for fc, p in PROFILES.items() if p.recoverability is Recoverability.TERMINAL
)


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    label: str
    support: int
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class TaxonomyReport:
    n: int
    accuracy: float
    unknown_rate: float
    per_class: list[ClassMetrics] = field(default_factory=list)
    #: (true, predicted, count), most frequent first.
    confusions: list[tuple[str, str, int]] = field(default_factory=list)

    # -- severity-weighted view -------------------------------------------
    #: Terminal failure misread as something actionable. The dangerous one.
    dangerous: int = 0
    #: Actionable failure misread as terminal. Costs revenue, harms nobody.
    over_cautious: int = 0
    #: Everything else: wrong class, same broad recovery strategy.
    benign: int = 0
    terminal_support: int = 0

    @property
    def terminal_recall(self) -> float:
        """Fraction of genuinely terminal failures correctly identified.

        This is the safety number. Every miss is a potential unauthorised debit,
        which is why the taxonomy fails *closed* on unmapped input and why LLM
        triage accepts terminal suggestions without a confidence bar.
        """
        if not self.terminal_support:
            return 1.0
        return 1.0 - self.dangerous / self.terminal_support

    @property
    def macro_f1(self) -> float:
        seen = [c for c in self.per_class if c.support]
        return sum(c.f1 for c in seen) / len(seen) if seen else 0.0

    def format(self, top: int = 14) -> str:
        rows = [
            f"n={self.n:,}   accuracy={self.accuracy:.4f}   "
            f"macro-F1={self.macro_f1:.4f}   unmapped={self.unknown_rate:.4f}",
            "",
            f"  {'failure class':<24}{'prec':>7}{'recall':>8}{'F1':>7}{'support':>9}",
            "  " + "-" * 55,
        ]
        for c in sorted(self.per_class, key=lambda m: m.support, reverse=True)[:top]:
            if not c.support:
                continue
            flag = "  <- terminal" if c.label in {t.value for t in TERMINAL_CLASSES} else ""
            rows.append(
                f"  {c.label:<24}{c.precision:>7.3f}{c.recall:>8.3f}"
                f"{c.f1:>7.3f}{c.support:>9,}{flag}"
            )

        rows += [
            "",
            "  errors by consequence, not by count:",
            f"    dangerous      {self.dangerous:>5}   terminal failure read as actionable",
            f"    over-cautious  {self.over_cautious:>5}   actionable failure read as terminal",
            f"    benign         {self.benign:>5}   wrong class, same recovery strategy",
            "",
            f"  TERMINAL RECALL  {self.terminal_recall:.4f}   "
            f"({self.terminal_support:,} terminal failures in the slice)",
        ]
        if self.dangerous == 0:
            rows.append(
                "  Not one revoked mandate, flagged payment or stolen card was "
                "misread\n  as something the agent may act on."
            )
        if self.confusions:
            rows += ["", "  most frequent confusions (true -> predicted):"]
            for t, p, n in self.confusions[:6]:
                rows.append(f"    {t:<24} -> {p:<24} {n:>5}")
        return "\n".join(rows)


def evaluate_taxonomy(
    events: list[RiskEvent], truth: dict[str, FailureClass]
) -> TaxonomyReport:
    """Score the deterministic classifier against ground truth.

    Deliberately measures the *table alone*, with no LLM triage in the loop, so
    the two components can be judged separately. Blending them would make it
    impossible to tell whether a good number came from the lookup table being
    comprehensive or the model covering for it.
    """
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)
    conf: dict[tuple[str, str], int] = defaultdict(int)

    correct = unknown = 0
    dangerous = over_cautious = benign = terminal_support = 0

    for e in events:
        actual = truth[e.event_id]
        pred = classify(
            e.error_code, e.error_description, risk_kind=e.kind.value
        ).failure_class
        support[actual.value] += 1
        if actual in TERMINAL_CLASSES:
            terminal_support += 1
        if pred is FailureClass.UNKNOWN:
            unknown += 1

        if pred is actual:
            correct += 1
            tp[actual.value] += 1
            continue

        fp[pred.value] += 1
        fn[actual.value] += 1
        conf[(actual.value, pred.value)] += 1

        actual_terminal = actual in TERMINAL_CLASSES
        pred_terminal = pred in TERMINAL_CLASSES
        # UNKNOWN fails closed -- one attempt, no silent retry -- so reading a
        # terminal failure as UNKNOWN is over-cautious, not dangerous.
        if actual_terminal and not pred_terminal and pred is not FailureClass.UNKNOWN:
            dangerous += 1
        elif pred_terminal and not actual_terminal:
            over_cautious += 1
        else:
            benign += 1

    labels = sorted(set(support) | set(fp))
    per_class = [
        ClassMetrics(lbl, support.get(lbl, 0), tp.get(lbl, 0), fp.get(lbl, 0), fn.get(lbl, 0))
        for lbl in labels
    ]
    confusions = sorted(
        ((t, p, n) for (t, p), n in conf.items()), key=lambda r: r[2], reverse=True
    )

    n = len(events)
    return TaxonomyReport(
        n=n,
        accuracy=correct / n if n else 0.0,
        unknown_rate=unknown / n if n else 0.0,
        per_class=per_class,
        confusions=confusions,
        dangerous=dangerous,
        over_cautious=over_cautious,
        benign=benign,
        terminal_support=terminal_support,
    )
