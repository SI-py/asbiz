"""Границы регламента и проверки Draft без сети."""
import asyncio

import pytest
from pydantic import ValidationError

from desk.escalation import Draft, Signals, atriage_many, needs_human, to_ticket
from desk.llm import LLMError
from tests.fakes import FakeLLM
from tests.test_hw2_escalation import GOOD, NONE, SOURCE, draft


@pytest.mark.parametrize('field,amount,expected', [
    ('refund', None, False), ('refund', 0, False),
    ('refund', 5000, False), ('refund', 5001, True),
    ('duplicate', None, False), ('duplicate', 15000, False),
    ('duplicate', 15001, True),
])
def test_money_boundaries(field, amount, expected):
    assert needs_human(Signals(**{**NONE, field: amount})) is expected


@pytest.mark.parametrize('field', [
    'fraud', 'threat', 'asks_human', 'tariff_pro', 'merchant_down', 'key_leak',
])
def test_each_boolean_is_enough(field):
    assert needs_human(Signals(**{**NONE, field: True}))


@pytest.mark.parametrize('changes', [
    {'quote': 'клиенту не доставили товар'}, {'payment_ids': []},
    {'payment_ids': ['P-123']}, {'severity': 6},
    {'signals': {**NONE, 'duplicate': -1}},
])
def test_draft_rejects_invalid_data(changes):
    with pytest.raises(ValidationError):
        Draft.model_validate({**GOOD, **changes}, context={'source': SOURCE})


def test_amount_is_not_a_refund_signal():
    d = Draft.model_validate({**GOOD, 'signals': NONE, 'needs_human': True},
                             context={'source': SOURCE})
    assert to_ticket(d).needs_human is False


def test_batch_keeps_order_and_other_results_after_error():
    def fail(_):
        raise LLMError('сбой')

    llm = FakeLLM([draft(), fail, draft(signals=NONE)])
    results = asyncio.run(atriage_many(llm, [SOURCE] * 3, concurrency=1))
    assert results[0].needs_human is True
    assert isinstance(results[1], LLMError)
    assert results[2].needs_human is False
