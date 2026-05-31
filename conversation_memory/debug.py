from __future__ import annotations

import json


def format_plan_debug(plan) -> str:
    return json.dumps(plan.debug, ensure_ascii=False, indent=2)


def print_plan_debug(plan) -> None:
    print(format_plan_debug(plan))
