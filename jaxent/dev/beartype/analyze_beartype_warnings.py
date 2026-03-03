#!/usr/bin/env python3
"""
Script to analyze @beartype warnings from pytest output.
Groups warnings by:
- Function being called
- Type of violation
- Test file/test that triggered it
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple
import sys


def parse_beartype_warnings(filepath: str) -> List[Dict[str, str]]:
    """Parse beartype warnings from pytest output file."""
    warnings = []
    current_test = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Extract test name
            if line and not line.startswith('<@beartype'):
                # This could be a test name line
                current_test = line
            
            # Extract beartype warning
            if line.startswith('<@beartype'):
                match = re.search(
                    r'<@beartype\((.*?)\) at (0x[0-9a-f]+)>:(\d+): UserWarning: Function (.*?)\(\) parameter (.*?) violates type hint (.*?), as (.*?)$',
                    line
                )
                if match:
                    warnings.append({
                        'function_decorated': match.group(1),
                        'address': match.group(2),
                        'line_num': match.group(3),
                        'function_name': match.group(4),
                        'parameter_info': match.group(5),
                        'expected_type': match.group(6),
                        'violation': match.group(7),
                        'test_context': current_test
                    })
    
    return warnings


def group_warnings(warnings: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group warnings by function being called."""
    grouped = defaultdict(list)
    for warning in warnings:
        key = warning['function_name']
        grouped[key].append(warning)
    return grouped


def group_by_test(warnings: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Group warnings by test file."""
    grouped = defaultdict(list)
    for warning in warnings:
        test = warning.get('test_context', 'Unknown')
        # Extract just the test file name
        if '::' in test:
            test_file = test.split('::')[0]
        else:
            test_file = test
        grouped[test_file].append(warning)
    return grouped


def extract_parameter_name(param_info: str) -> str:
    """Extract parameter name from parameter info string."""
    # Format is usually: parameter_name="value..."
    if '=' in param_info:
        return param_info.split('=')[0].strip('"')
    return param_info


def print_summary_report(warnings: List[Dict[str, str]]):
    """Print a comprehensive summary report."""
    
    print("=" * 80)
    print(" BEARTYPE WARNINGS ANALYSIS REPORT")
    print("=" * 80)
    print(f"\nTotal warnings found: {len(warnings)}\n")
    
    # Group by function
    by_function = group_warnings(warnings)
    print(f"Number of unique functions with warnings: {len(by_function)}\n")
    
    print("-" * 80)
    print("WARNINGS BY FUNCTION (sorted by count)")
    print("-" * 80)
    
    sorted_funcs = sorted(by_function.items(), key=lambda x: len(x[1]), reverse=True)
    for func_name, func_warnings in sorted_funcs:
        print(f"\n{func_name}: {len(func_warnings)} warnings")
        
        # Get unique parameter names
        params = set()
        expected_types = set()
        violations = set()
        
        for w in func_warnings:
            param_name = extract_parameter_name(w['parameter_info'])
            params.add(param_name)
            expected_types.add(w['expected_type'])
            violations.add(w['violation'])
        
        print(f"  Parameters: {', '.join(sorted(params))}")
        print(f"  Expected type(s): {', '.join(sorted(expected_types))}")
        
        # Show a sample violation
        if violations:
            sample_violation = list(violations)[0]
            if len(sample_violation) > 100:
                sample_violation = sample_violation[:100] + "..."
            print(f"  Sample violation: {sample_violation}")
    
    print("\n" + "-" * 80)
    print("WARNINGS BY TEST FILE")
    print("-" * 80)
    
    by_test = group_by_test(warnings)
    sorted_tests = sorted(by_test.items(), key=lambda x: len(x[1]), reverse=True)
    
    for test_file, test_warnings in sorted_tests:
        print(f"\n{test_file}: {len(test_warnings)} warnings")
        
        # Get unique functions in this test
        funcs_in_test = set(w['function_name'] for w in test_warnings)
        print(f"  Functions: {', '.join(sorted(funcs_in_test)[:5])}", end="")
        if len(funcs_in_test) > 5:
            print(f" ... and {len(funcs_in_test) - 5} more")
        else:
            print()
    
    print("\n" + "-" * 80)
    print("UNIQUE VIOLATION PATTERNS")
    print("-" * 80)
    
    # Group by expected type
    by_expected_type = defaultdict(list)
    for w in warnings:
        by_expected_type[w['expected_type']].append(w)
    
    print(f"\nUnique expected types: {len(by_expected_type)}")
    for expected_type, type_warnings in sorted(by_expected_type.items(), 
                                               key=lambda x: len(x[1]), 
                                               reverse=True):
        print(f"\n  {expected_type}: {len(type_warnings)} warnings")
        funcs = set(w['function_name'] for w in type_warnings)
        print(f"    Affected functions ({len(funcs)}): {', '.join(sorted(funcs)[:3])}", end="")
        if len(funcs) > 3:
            print(f" ... and {len(funcs) - 3} more")
        else:
            print()
    
    print("\n" + "=" * 80)


def main():
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'pytest_out.txt'
    
    print(f"Analyzing beartype warnings from: {filepath}\n")
    
    warnings = parse_beartype_warnings(filepath)
    
    if not warnings:
        print("No beartype warnings found!")
        return
    
    print_summary_report(warnings)
    
    # Optionally save detailed results to a file
    output_file = 'beartype_warnings_detailed.txt'
    with open(output_file, 'w') as f:
        f.write("DETAILED BEARTYPE WARNINGS\n")
        f.write("=" * 80 + "\n\n")
        
        by_function = group_warnings(warnings)
        for func_name, func_warnings in sorted(by_function.items()):
            f.write(f"\nFunction: {func_name}\n")
            f.write(f"Total warnings: {len(func_warnings)}\n")
            f.write("-" * 80 + "\n")
            
            # Show first few unique violations
            seen_violations = set()
            for w in func_warnings[:10]:  # Limit to first 10
                violation_key = (
                    w['parameter_info'], 
                    w['expected_type'],
                    w['violation']
                )
                if violation_key not in seen_violations:
                    seen_violations.add(violation_key)
                    f.write(f"\n  Test: {w['test_context']}\n")
                    f.write(f"  Parameter: {w['parameter_info']}\n")
                    f.write(f"  Expected: {w['expected_type']}\n")
                    f.write(f"  Violation: {w['violation']}\n")
            
            if len(func_warnings) > 10:
                f.write(f"\n  ... and {len(func_warnings) - 10} more warnings\n")
            f.write("\n")
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
