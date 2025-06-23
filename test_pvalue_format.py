#!/usr/bin/env python3
"""
Test script to verify p-value formatting function
"""

def format_p_value(p):
    """Format p-value with scientific notation"""
    p_rounded = round(p, 2)
    if p > 0.05:
        return {'p_value': p_rounded, 'notation': 'N.S.'}
    elif p > 0.01:
        return {'p_value': p_rounded, 'notation': '*'}
    elif p > 0.001:
        return {'p_value': p_rounded, 'notation': '**'}
    else:
        return {'p_value': p_rounded, 'notation': '***'}

# Test cases
test_values = [
    0.8,     # Should be N.S.
    0.06,    # Should be N.S.
    0.05,    # Should be N.S.
    0.049,   # Should be *
    0.02,    # Should be *
    0.01,    # Should be *
    0.009,   # Should be **
    0.005,   # Should be **
    0.001,   # Should be **
    0.0009,  # Should be ***
    0.0001   # Should be ***
]

print("Testing p-value formatting:")
print("P-value\t\tRounded\t\tNotation")
print("-" * 40)

for p in test_values:
    result = format_p_value(p)
    print(f"{p:.4f}\t\t{result['p_value']:.2f}\t\t{result['notation']}")
