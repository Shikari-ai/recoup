"""Churn cost: what pursuing a receivable does to the relationship behind it.

The expected-value calculation this project shipped with is blind to annoyance.
It prices the send (a WhatsApp costs 55 paise) but not the consequence of the
send, so on a large receivable it will happily spend a customer's patience
because patience has no line item. That is the classic failure mode of a
collections system: it optimises the receivable and expenses the relationship.

The fix is to give annoyance a price, in the same units as everything else::

    EV_adjusted = P(recover) x amount_at_risk - action_cost - P(churn) x LTV

Three things make this honest rather than decorative:

**Only contact incurs churn.** Switching rails, waiting and stopping are
invisible to the customer, so their churn term is exactly zero. A model that
charged churn for a silent retry would be describing a different world.

**Churn compounds with contact, it does not accumulate linearly.** The fourth
message in a week is not four times as irritating as the first; it is the one
that gets you blocked. Hence a geometric term in the count of recent messages,
defaulting to 1.5x per message. That constant is a stated assumption, not a
measurement -- see the caveat below.

**LTV defaults to zero, and zero means "not known".** A merchant who has not
supplied lifetime value gets ``P(churn) x 0 == 0`` and therefore precisely the
behaviour this engine had before churn existed. The feature cannot silently
change decisions for anyone who has not opted into it by supplying the data,
which is also why adding it does not move any published figure.

**Caveat, stated once and loudly.** These base rates are assumptions. Nobody
here has measured the churn probability of an SMS. What the machinery buys is
not a correct number but a *place to put* the number: a merchant with real
retention data can drop it into the pack and the engine will spend their
customers' patience at whatever rate they say it is worth. Treat the shipped
defaults as a shape, and see ``scripts/churn_sensitivity.py`` for how much the
decisions move when the shape is wrong.
"""

from __future__ import annotations

from .domain import COMMS_ACTIONS, Action, Channel, RiskEvent

#: Probability that a single message of each kind is the one that loses the
#: customer, before any fatigue multiplier. Ordered by intrusiveness rather than
#: by cost: an automated voice call is cheap to place and expensive to receive.
#:
#: Email is an order of magnitude below SMS because it is trivially ignored --
#: the cost of an unwanted email is a deleted email. A voice call interrupts.
CHURN_BASE: dict[Channel, float] = {
    Channel.VOICE: 0.020,
    Channel.WHATSAPP: 0.005,
    Channel.SMS: 0.002,
    Channel.EMAIL: 0.001,
    Channel.NONE: 0.0,
}

#: Multiplier applied per message already sent to this customer recently.
DEFAULT_CHURN_GROWTH = 1.5

#: Ceiling on the fatigue exponent. Without it, a customer with 40 messages on
#: record produces 1.5**40 -- about ten million -- and the churn term dwarfs
#: every receivable in the book, so the engine stops doing anything at all for
#: anyone. The comms frequency cap should make counts this high unreachable, but
#: "should" is doing load-bearing work in that sentence and this is cheaper than
#: finding out. Ten messages already puts the multiplier near 58x.
MAX_FATIGUE_EXPONENT = 10


def recent_contacts(event: RiskEvent) -> int:
    """How many messages this customer has recently received.

    Two counters could answer this and they measure different things:
    ``customer.comms_sent_7d`` is every message from this merchant in the
    rolling window, across all of that customer's receivables, while
    ``event.comms_taken`` counts only the ones sent chasing *this* one.

    Annoyance is a property of the person, not of the invoice, so the
    customer-level count is the right one. The event-level count is taken as a
    floor rather than ignored: if a caller has populated one and not the other,
    the conservative reading is the larger, and under-counting here means
    under-pricing the harm.
    """
    return max(event.customer.comms_sent_7d, event.comms_taken)


def churn_probability(
    event: RiskEvent,
    action: Action,
    *,
    growth: float = DEFAULT_CHURN_GROWTH,
    base: dict[Channel, float] | None = None,
) -> float:
    """P(this action is the one that loses the customer).

    Zero for anything the customer cannot perceive. Geometric in the number of
    messages they have already had, capped, and clamped to a probability.
    """
    if action.kind not in COMMS_ACTIONS:
        return 0.0
    table = base if base is not None else CHURN_BASE
    p0 = table.get(action.channel, 0.0)
    if p0 <= 0.0:
        return 0.0
    n = min(recent_contacts(event), MAX_FATIGUE_EXPONENT)
    return min(1.0, p0 * (growth ** n))


def churn_cost_paise(
    event: RiskEvent,
    action: Action,
    *,
    growth: float = DEFAULT_CHURN_GROWTH,
    base: dict[Channel, float] | None = None,
) -> int:
    """The churn term of the EV equation, in paise.

    Returns 0 whenever lifetime value is unknown, which is the default. That is
    what keeps this change inert for every caller that has not supplied LTV --
    including every existing test and every published figure.
    """
    ltv = event.customer.ltv_paise
    if ltv <= 0:
        return 0
    p = churn_probability(event, action, growth=growth, base=base)
    if p <= 0.0:
        return 0
    return int(round(p * ltv))
