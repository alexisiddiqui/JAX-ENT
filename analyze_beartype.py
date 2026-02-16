import re
from collections import Counter
import sys

def analyze_beartype_log(file_path):
    exception_counts = Counter()
    unique_issues = {}
    
    # Regex to find Beartype related lines
    # Example: /path/to/file.py:30: BeartypeClawDecorWarning: ...
    # Example: beartype.roar.BeartypeDecorHintPep3119Exception: ...
    
    re_warning = re.compile(r'([\w/.-]+\.py:\d+): (Beartype\w+): (.*)')
    re_exception = re.compile(r'(beartype\.roar\.\w+): (.*)')
    
    current_pytest_test = None
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Track current test (optional but helpful)
                if line.startswith('jaxent/tests/') and ('[' in line or line.endswith('.')):
                    current_pytest_test = line.split()[0]
                
                # Match warnings with file info
                match_w = re_warning.search(line)
                if match_w:
                    file_info, exc_type, message = match_w.groups()
                    key = (exc_type, message.strip())
                    exception_counts[exc_type] += 1
                    if key not in unique_issues:
                        unique_issues[key] = {'count': 0, 'files': Counter()}
                    unique_issues[key]['count'] += 1
                    unique_issues[key]['files'][file_info] += 1
                    continue
                
                # Match exceptions
                match_e = re_exception.search(line)
                if match_e:
                    exc_type, message = match_e.groups()
                    # Clean message from memory addresses
                    clean_msg = re.sub(r'at 0x[0-9a-fA-F]+', 'at 0x...', message.strip())
                    key = (exc_type, clean_msg)
                    exception_counts[exc_type] += 1
                    if key not in unique_issues:
                        unique_issues[key] = {'count': 0, 'files': Counter()}
                    unique_issues[key]['count'] += 1
                    # Try to find the last file mentioned if possible, or just mark as unknown
                    unique_issues[key]['files']['unknown'] += 1
                    continue

        print("=== Beartype Analysis Report ===")
        print(f"Total Beartype issues found: {sum(exception_counts.values())}")
        print("\n--- Issue Type Summary ---")
        for exc_type, count in exception_counts.most_common():
            print(f"{exc_type}: {count}")
            
        print("\n--- Detailed Unique Issues ---")
        # Sort by count
        sorted_issues = sorted(unique_issues.items(), key=lambda x: x[1]['count'], reverse=True)
        
        for (exc_type, message), data in sorted_issues:
            print(f"\n[{data['count']}] {exc_type}")
            print(f"Message: {message[:500]}...")
            print("Top files/locations:")
            for file_info, f_count in data['files'].most_common(5):
                print(f"  - {file_info} ({f_count})")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    log_file = "beartype_log2.txt"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    analyze_beartype_log(log_file)
