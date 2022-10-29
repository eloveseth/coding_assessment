import re

from src.utils import extract_pay_type, extract_pay_amt


def test_extract_pay_type():
    pay_rate = '2012.00 PER_HOUR'
    pay_type = re.sub('[0-9\.\s]', '', pay_rate)

    assert pay_type == 'PER_HOUR'


def test_extract_pay_amt():
    pay_rate = '2012.00 PER_HOUR'
    pay_amt = re.sub('[\s_a-zA-Z]', '', pay_rate)

    assert pay_amt == '2012.00'
