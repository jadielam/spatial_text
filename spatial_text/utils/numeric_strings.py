from typing import List


def is_numeric(a: str):
    try:
        float(a)
        return True
    except ValueError:
        return False


def generate_numeric_candidates(a: str) -> List[str]:
    a_f = float(a)
    if a_f == 0:
        return ['0', '0.0', '0.00']
    if int(a_f) == a_f:
        return [f'{a_f:.0f}', f'{a_f:.1f}', f'{a_f:.2f}']
    if a_f > 0 and a_f < 1:
        return [
            f'{a_f:.1f}',
            f'{a_f:.2f}',
            f'{a_f:.1f}'.lstrip('0'),
            f'{a_f:.2f}'.lstrip('0'),
        ]
    return [f'{a_f:.1f}', f'{a_f:.2f}']
