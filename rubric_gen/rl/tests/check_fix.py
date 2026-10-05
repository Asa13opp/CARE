import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from rl.schema import parse_rubric, validate_rubric
from rl.rewards import reward_format, reward_trans
from rl.tests.test_schema import GOOD

junk = "aget\n" + json.dumps(GOOD) + "\ntrailing text here"
obj, err = parse_rubric(junk)
T = ["advice", "questioning", "emotional_expression", "paraphrasing"]
lines = [
    "parse_err=%r" % err,
    "reward_format=%r" % reward_format(junk),
    "reward_trans=%r" % reward_trans(junk, T),
    "validate_extracted=%r" % validate_rubric(obj),
]
out = "\n".join(lines) + "\n"
open("/tmp/fixcheck.txt", "w").write(out)
print(out)
