"""Synthetic at-risk event generator.

Produces a stream of ``RiskEvent`` shaped like a real Razorpay merchant's
failure feed: the rail mix Indian consumers actually use, failure reasons
distributed per rail, ticket sizes drawn per merchant archetype, and issuers
that go down in bursts.

Four archetypes, because the recovery problem genuinely differs between them
and a single blended population would hide that:

* ``subscription``  -- low ticket, mandate rails, insufficient funds dominates
* ``d2c``           -- mid ticket, UPI-heavy, abandonment and auth failures
* ``b2b``           -- high ticket, invoices and NACH, long horizons
* ``edtech``        -- high ticket EMI, card-heavy, limit and auth declines

Roughly 3% of events carry an error string that is deliberately *not* in the
taxonomy. Real gateways ship new reason codes without warning, and a classifier
that has never met an unmapped input has not been tested. Those flow to the
UNKNOWN path and on to LLM triage.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..domain import (
    Channel,
    CustomerContext,
    FailureClass,
    Rail,
    RiskEvent,
    RiskKind,
)
from .world import World

ISSUERS = [
    "HDFC", "ICICI", "SBI", "AXIS", "KOTAK",
    "PNB", "BOB", "YESBANK", "IDFC", "INDUSIND", "PAYTM", "AUBANK",
]

#: Representative gateway error codes per class, so the taxonomy is exercised
#: through the same surface a real integration would present.
CODE_FOR: dict[FailureClass, list[str]] = {
    FailureClass.INSUFFICIENT_FUNDS: ["insufficient_funds", "BANK_INSUFFICIENT_BALANCE", "ISO_51"],
    FailureClass.ISSUER_DOWN: ["issuer_down", "bank_down", "payment_issuer_unavailable"],
    FailureClass.GATEWAY_ERROR: ["gateway_error", "SERVER_ERROR"],
    FailureClass.NETWORK_TIMEOUT: ["payment_timeout", "gateway_timeout"],
    FailureClass.RATE_LIMITED: ["rate_limit_exceeded"],
    FailureClass.VELOCITY_LIMIT: ["payment_limit_exceeded", "ISO_61"],
    FailureClass.AUTH_FAILED: ["authentication_failed", "incorrect_otp", "3ds_failed"],
    FailureClass.COLLECT_EXPIRED: ["collect_request_expired", "upi_collect_expired"],
    FailureClass.ABANDONED: ["payment_cancelled_by_user", "checkout_abandoned"],
    FailureClass.MANDATE_PAUSED: ["mandate_paused", "subscription_paused"],
    FailureClass.INVOICE_UNPAID: ["invoice_overdue"],
    FailureClass.CARD_EXPIRED: ["card_expired", "ISO_54"],
    FailureClass.TOKEN_EXPIRED: ["token_expired"],
    FailureClass.INVALID_INSTRUMENT: ["invalid_vpa", "invalid_card"],
    FailureClass.ACCOUNT_CLOSED: ["account_closed", "ISO_46"],
    FailureClass.INTERNATIONAL_BLOCKED: ["international_transaction_not_allowed"],
    FailureClass.MANDATE_REVOKED: ["mandate_revoked", "subscription_cancelled"],
    FailureClass.RISK_DECLINED: ["risk_threshold_exceeded"],
    FailureClass.SUSPECTED_FRAUD: ["stolen_card", "ISO_43"],
    FailureClass.DO_NOT_HONOUR: ["do_not_honour", "payment_declined_by_bank", "ISO_05"],
}

#: Error strings the taxonomy has never seen. Modelled on how real gateways
#: actually ship changes: new vendor prefixes, localised text, opaque codes.
NOVEL_CODES: list[tuple[str, str, FailureClass]] = [
    ("PSP_ERR_7734", "Remitter account frozen by issuer directive", FailureClass.ACCOUNT_CLOSED),
    ("NPCI_XC_09", "Beneficiary PSP unreachable, retry advised", FailureClass.ISSUER_DOWN),
    ("ACQ_DENY_2201", "Balance below required threshold", FailureClass.INSUFFICIENT_FUNDS),
    ("UPI_MANDATE_STATE_INVALID", "Umandate in suspended state", FailureClass.MANDATE_PAUSED),
    ("CARD_VAULT_MISS", "Stored credential no longer resolvable", FailureClass.TOKEN_EXPIRED),
    ("BANK_MSG_04", "Kripya baad mein prayaas karein", FailureClass.ISSUER_DOWN),
]


@dataclass(frozen=True, slots=True)
class Archetype:
    name: str
    #: Log-normal ticket size parameters, in rupees.
    amount_mu: float
    amount_sigma: float
    rail_weights: dict[Rail, float]
    kind_weights: dict[RiskKind, float]
    #: Multiplicative tilt on the global failure-class mix.
    class_tilt: dict[FailureClass, float] = field(default_factory=dict)
    deadline_days: int | None = None


ARCHETYPES: list[Archetype] = [
    Archetype(
        name="subscription",
        amount_mu=math.log(499),
        amount_sigma=0.55,
        rail_weights={Rail.UPI_AUTOPAY: 0.52, Rail.CARD_TOKEN: 0.30, Rail.EMANDATE_NACH: 0.18},
        kind_weights={RiskKind.SUBSCRIPTION_CHARGE_FAILED: 0.7, RiskKind.MANDATE_DEBIT_FAILED: 0.3},
        class_tilt={
            FailureClass.INSUFFICIENT_FUNDS: 2.4,
            FailureClass.MANDATE_REVOKED: 1.8,
            FailureClass.MANDATE_PAUSED: 2.0,
            FailureClass.TOKEN_EXPIRED: 1.6,
            FailureClass.ABANDONED: 0.1,
        },
        deadline_days=21,
    ),
    Archetype(
        name="d2c",
        amount_mu=math.log(1450),
        amount_sigma=0.85,
        rail_weights={
            Rail.UPI_COLLECT: 0.58, Rail.CARD: 0.22,
            Rail.NETBANKING: 0.12, Rail.WALLET: 0.08,
        },
        kind_weights={RiskKind.PAYMENT_FAILED: 0.62, RiskKind.CHECKOUT_ABANDONED: 0.38},
        class_tilt={
            FailureClass.ABANDONED: 3.0,
            FailureClass.COLLECT_EXPIRED: 2.2,
            FailureClass.AUTH_FAILED: 1.4,
        },
        deadline_days=7,
    ),
    Archetype(
        name="b2b",
        amount_mu=math.log(68000),
        amount_sigma=1.05,
        rail_weights={Rail.EMANDATE_NACH: 0.42, Rail.NETBANKING: 0.34, Rail.PAYMENT_LINK: 0.24},
        kind_weights={RiskKind.INVOICE_OVERDUE: 0.72, RiskKind.MANDATE_DEBIT_FAILED: 0.28},
        class_tilt={
            FailureClass.INVOICE_UNPAID: 5.0,
            FailureClass.INSUFFICIENT_FUNDS: 1.5,
            FailureClass.ABANDONED: 0.05,
        },
        deadline_days=45,
    ),
    Archetype(
        name="edtech",
        amount_mu=math.log(11500),
        amount_sigma=0.75,
        rail_weights={Rail.CARD: 0.44, Rail.UPI_COLLECT: 0.30, Rail.NETBANKING: 0.16, Rail.CARD_TOKEN: 0.10},
        kind_weights={RiskKind.PAYMENT_FAILED: 0.70, RiskKind.SUBSCRIPTION_CHARGE_FAILED: 0.30},
        class_tilt={
            FailureClass.VELOCITY_LIMIT: 2.6,
            FailureClass.AUTH_FAILED: 1.8,
            FailureClass.DO_NOT_HONOUR: 1.6,
            FailureClass.CARD_EXPIRED: 1.5,
        },
        deadline_days=30,
    ),
]

#: Global prior over failure classes before archetype tilt.
BASE_CLASS_MIX: dict[FailureClass, float] = {
    FailureClass.INSUFFICIENT_FUNDS: 0.205,
    FailureClass.AUTH_FAILED: 0.130,
    FailureClass.ABANDONED: 0.110,
    FailureClass.DO_NOT_HONOUR: 0.095,
    FailureClass.ISSUER_DOWN: 0.085,
    FailureClass.COLLECT_EXPIRED: 0.075,
    FailureClass.GATEWAY_ERROR: 0.055,
    FailureClass.NETWORK_TIMEOUT: 0.045,
    FailureClass.CARD_EXPIRED: 0.040,
    FailureClass.VELOCITY_LIMIT: 0.035,
    FailureClass.INVOICE_UNPAID: 0.030,
    FailureClass.MANDATE_PAUSED: 0.025,
    FailureClass.TOKEN_EXPIRED: 0.022,
    FailureClass.INVALID_INSTRUMENT: 0.018,
    FailureClass.MANDATE_REVOKED: 0.010,
    FailureClass.RISK_DECLINED: 0.008,
    FailureClass.INTERNATIONAL_BLOCKED: 0.006,
    FailureClass.ACCOUNT_CLOSED: 0.003,
    FailureClass.SUSPECTED_FRAUD: 0.002,
    FailureClass.RATE_LIMITED: 0.001,
}


@dataclass
class ScenarioConfig:
    n_events: int = 5000
    days: int = 45
    seed: int = 42
    n_customers: int = 1800
    novel_code_rate: float = 0.03
    dnd_rate: float = 0.12
    start: datetime | None = None


def _weighted(rng: random.Random, weights: dict) -> object:
    total = sum(weights.values())
    r = rng.random() * total
    acc = 0.0
    for k, w in weights.items():
        acc += w
        if r <= acc:
            return k
    return next(iter(weights))


def _make_customer(rng: random.Random, cid: str, dnd_rate: float) -> CustomerContext:
    seen = rng.choices([0, 1, 2, 3, 5, 8, 14, 25], weights=[18, 16, 15, 13, 14, 12, 8, 4])[0]
    fails = min(seen, rng.randint(0, max(1, seen // 2)))
    known: list[Rail] = []
    if seen > 0:
        pool = [Rail.UPI_COLLECT, Rail.CARD, Rail.NETBANKING, Rail.UPI_AUTOPAY, Rail.WALLET]
        known = rng.sample(pool, k=min(len(pool), rng.choices([1, 2, 3], weights=[6, 3, 1])[0]))

    channels: list[Channel] = [Channel.SMS]
    if rng.random() < 0.74:
        channels.append(Channel.WHATSAPP)
    if rng.random() < 0.66:
        channels.append(Channel.EMAIL)
    if rng.random() < 0.10:
        channels.append(Channel.VOICE)

    return CustomerContext(
        customer_id=cid,
        prior_successes=seen - fails,
        prior_failures=fails,
        known_rails=tuple(known),
        dnd_registered=rng.random() < dnd_rate,
        contactable=tuple(channels),
        locale=rng.choices(["en_IN", "hi_IN", "hinglish"], weights=[52, 24, 24])[0],
        comms_sent_7d=rng.choices([0, 0, 0, 1, 2], weights=[60, 15, 10, 10, 5])[0],
    )


def generate(config: ScenarioConfig, world: World | None = None) -> tuple[list[RiskEvent], World, dict[str, FailureClass]]:
    """Build a scenario.

    Returns the events, the latent world that will resolve their outcomes, and
    the ``event_id -> true FailureClass`` map. That map exists so the taxonomy
    can be scored for accuracy independently of the recovery policy -- a
    classifier that is quietly 60% accurate would poison every downstream
    number, and you would never see it by looking at rupees recovered alone.
    """
    rng = random.Random(config.seed)
    start = config.start or datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    w = world or World(seed=config.seed, start=start, days=config.days)

    # Sorted, not a set: Rail is a str Enum, so set iteration order is
    # hash-randomised per process. Feeding that order into the outage RNG
    # made the whole backtest non-reproducible across runs at a fixed seed.
    rails_used = sorted({r for a in ARCHETYPES for r in a.rail_weights}, key=lambda r: r.value)
    w.seed_outages(ISSUERS, rails_used, config.days)

    customers = [
        _make_customer(random.Random(f"{config.seed}:c:{i}"), f"cust_{i:05d}", config.dnd_rate)
        for i in range(config.n_customers)
    ]

    events: list[RiskEvent] = []
    truth: dict[str, FailureClass] = {}

    for i in range(config.n_events):
        arch = rng.choices(ARCHETYPES, weights=[0.34, 0.36, 0.12, 0.18])[0]
        cust = rng.choice(customers)
        rail: Rail = _weighted(rng, arch.rail_weights)  # type: ignore[assignment]
        kind: RiskKind = _weighted(rng, arch.kind_weights)  # type: ignore[assignment]

        mix = {
            fc: p * arch.class_tilt.get(fc, 1.0)
            for fc, p in BASE_CLASS_MIX.items()
        }
        # Rails constrain which failures are even possible. A UPI collect
        # request cannot expire a card, and a card cannot have its VPA rejected.
        if rail in (Rail.UPI_COLLECT, Rail.UPI_AUTOPAY):
            for fc in (FailureClass.CARD_EXPIRED, FailureClass.TOKEN_EXPIRED,
                       FailureClass.INTERNATIONAL_BLOCKED):
                mix[fc] = 0.0
        if rail in (Rail.CARD, Rail.CARD_TOKEN):
            mix[FailureClass.COLLECT_EXPIRED] = 0.0
        if rail not in (Rail.UPI_AUTOPAY, Rail.EMANDATE_NACH, Rail.CARD_TOKEN):
            mix[FailureClass.MANDATE_REVOKED] = 0.0
            mix[FailureClass.MANDATE_PAUSED] = 0.0
        if kind is RiskKind.CHECKOUT_ABANDONED:
            mix = {FailureClass.ABANDONED: 1.0}
        if kind is RiskKind.INVOICE_OVERDUE:
            mix = {FailureClass.INVOICE_UNPAID: 1.0}

        fc: FailureClass = _weighted(rng, mix)  # type: ignore[assignment]

        amount = int(round(rng.lognormvariate(arch.amount_mu, arch.amount_sigma)))
        amount = max(49, min(amount, 900000))
        amount_paise = amount * 100

        occurred = start + timedelta(minutes=rng.uniform(0, config.days * 24 * 60))
        issuer = rng.choice(ISSUERS)

        # Bias reality toward coherence: if the issuer really is down at this
        # moment, an ISSUER_DOWN classification is far more likely. Without
        # this the health monitor would be learning from noise.
        if w.is_down(issuer, rail, occurred) and rng.random() < 0.75:
            fc = FailureClass.ISSUER_DOWN

        if rng.random() < config.novel_code_rate:
            code, desc, fc = rng.choice(NOVEL_CODES)
        else:
            code = rng.choice(CODE_FOR.get(fc, ["payment_failed"]))
            desc = code.replace("_", " ").lower()

        deadline = (
            occurred + timedelta(days=arch.deadline_days) if arch.deadline_days else None
        )

        eid = f"evt_{i:06d}"
        truth[eid] = fc
        events.append(
            RiskEvent(
                event_id=eid,
                merchant_id=f"mch_{arch.name}",
                kind=kind,
                amount_paise=amount_paise,
                rail=rail,
                occurred_at=occurred,
                customer=cust,
                error_code=code,
                error_description=desc,
                issuer=issuer,
                attempt_no=0,
                deadline=deadline,
                metadata={
                    "archetype": arch.name,
                    "card_scheme": rng.choice(["visa", "mastercard", "rupay"])
                    if rail in (Rail.CARD, Rail.CARD_TOKEN)
                    else None,
                    "instrument_id": f"inst_{cust.customer_id}_{rail.value}",
                },
            )
        )

    events.sort(key=lambda e: e.occurred_at)
    return events, w, truth
