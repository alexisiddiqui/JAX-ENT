"""
python code_metrics.py <directory1> <directory2> ... --output <output_directory> --names <name1> <name2> ...
This script analyzes Python files in the specified directories and generates histograms for various code metrics.

python /home/alexi/Documents/JAX-ENT/notebooks/quick_experiments/interpret_code/code_metrics.py \
    /home/alexi/Documents/JAX-ENT/jaxent/ \
    /home/alexi/Documents/HDXer/HDXer \
    --output /home/alexi/Documents/JAX-ENT/notebooks/quick_experiments/interpret_code/plots \
    --names "jaxENT" "HDXer"
"""

#!/usr/bin/env python3
import argparse
import ast
import io
import tokenize
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def count_lines_in_string(text):
    """Count the number of non-empty lines in a string."""
    return len([line for line in text.strip().split("\n") if line.strip()])


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor that calculates complexity metrics."""

    def __init__(self):
        self.cyclomatic_complexity = 1  # Start at 1
        self.cognitive_complexity = 0
        self.returns = 0
        self.current_nested_level = 0
        self.max_nested_level = 0
        self.nesting_increments = set()  # For cognitive complexity

    def visit_If(self, node):
        self.cyclomatic_complexity += 1  # Each 'if' adds a branch

        # Cognitive complexity: +1 for if, +nesting for nested if
        if id(node) not in self.nesting_increments:
            self.cognitive_complexity += 1 + self.current_nested_level
            self.nesting_increments.add(id(node))

        # Track nesting level
        self.current_nested_level += 1
        self.max_nested_level = max(self.max_nested_level, self.current_nested_level)

        # Visit children
        self.generic_visit(node)

        # Restore nesting level
        self.current_nested_level -= 1

    def visit_For(self, node):
        self.cyclomatic_complexity += 1  # Each 'for' adds a branch

        # Cognitive complexity: +1 for loop, +nesting for nested loop
        if id(node) not in self.nesting_increments:
            self.cognitive_complexity += 1 + self.current_nested_level
            self.nesting_increments.add(id(node))

        # Track nesting level
        self.current_nested_level += 1
        self.max_nested_level = max(self.max_nested_level, self.current_nested_level)

        # Visit children
        self.generic_visit(node)

        # Restore nesting level
        self.current_nested_level -= 1

    def visit_While(self, node):
        self.cyclomatic_complexity += 1  # Each 'while' adds a branch

        # Cognitive complexity: +1 for loop, +nesting for nested loop
        if id(node) not in self.nesting_increments:
            self.cognitive_complexity += 1 + self.current_nested_level
            self.nesting_increments.add(id(node))

        # Track nesting level
        self.current_nested_level += 1
        self.max_nested_level = max(self.max_nested_level, self.current_nested_level)

        # Visit children
        self.generic_visit(node)

        # Restore nesting level
        self.current_nested_level -= 1

    def visit_Try(self, node):
        self.cyclomatic_complexity += len(node.handlers)  # Each except handler adds a branch

        # Cognitive complexity: +1 for try
        if id(node) not in self.nesting_increments:
            self.cognitive_complexity += 1
            self.nesting_increments.add(id(node))

        # Track nesting level
        self.current_nested_level += 1
        self.max_nested_level = max(self.max_nested_level, self.current_nested_level)

        # Visit children
        self.generic_visit(node)

        # Restore nesting level
        self.current_nested_level -= 1

    def visit_With(self, node):
        # Track nesting level
        self.current_nested_level += 1
        self.max_nested_level = max(self.max_nested_level, self.current_nested_level)

        # Visit children
        self.generic_visit(node)

        # Restore nesting level
        self.current_nested_level -= 1

    def visit_BoolOp(self, node):
        # Cognitive complexity: +1 for each boolean operator (and, or)
        if isinstance(node.op, (ast.And, ast.Or)):
            self.cognitive_complexity += 1
        self.generic_visit(node)

    def visit_Return(self, node):
        self.returns += 1
        self.generic_visit(node)


def has_type_annotations(node):
    """Check if a function node has type annotations."""
    has_return_annotation = node.returns is not None
    arg_annotations = sum(1 for arg in node.args.args if arg.annotation is not None)
    total_args = len(node.args.args)

    # Calculate percentage of arguments with annotations
    arg_annotation_pct = arg_annotations / total_args if total_args > 0 else 0

    # Function has type hints if return is annotated or >50% of args are annotated
    return has_return_annotation or (arg_annotation_pct > 0.5)


def analyze_file(file_path):
    """Analyze a Python file for various code metrics."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return None

        # Parse AST
        tree = ast.parse(content)

        # Extract functions and their metrics
        function_metrics = []
        annotated_functions = 0
        total_functions = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1

                # Calculate function lines of code
                func_start = node.lineno
                func_end = node.end_lineno

                # Get docstring if it exists
                docstring = ast.get_docstring(node) or ""
                docstring_lines = count_lines_in_string(docstring) if docstring else 0

                # Collect variable names
                var_names = set()
                for n in ast.walk(node):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        var_names.add(n.id)
                    elif isinstance(n, ast.arg):
                        var_names.add(n.arg)

                var_lengths = [len(name) for name in var_names]

                # Calculate function LOC excluding docstring
                func_loc = func_end - func_start + 1

                # Calculate complexity metrics
                complexity_visitor = ComplexityVisitor()
                complexity_visitor.visit(node)

                # Check for type annotations
                if has_type_annotations(node):
                    annotated_functions += 1

                function_metrics.append(
                    {
                        "name": node.name,
                        "loc": func_loc,
                        "docstring_lines": docstring_lines,
                        "var_lengths": var_lengths,
                        "cyclomatic_complexity": complexity_visitor.cyclomatic_complexity,
                        "cognitive_complexity": complexity_visitor.cognitive_complexity,
                        "max_nested_level": complexity_visitor.max_nested_level,
                        "return_count": complexity_visitor.returns,
                    }
                )

        # Calculate type hint usage percentage
        type_hint_pct = (annotated_functions / total_functions) * 100 if total_functions > 0 else 0

        # Count comments, code, and docstrings
        comment_lines = 0
        total_lines = len(content.splitlines())
        docstring_lines = 0

        # Count docstrings
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node) or ""
                if docstring:
                    docstring_lines += count_lines_in_string(docstring)

        # Count comments using tokenize
        with io.BytesIO(content.encode("utf-8")) as f:
            for token in tokenize.tokenize(f.readline):
                if token.type == tokenize.COMMENT:
                    comment_lines += 1

        # Estimate code lines
        code_lines = total_lines - docstring_lines

        return {
            "comment_lines": comment_lines,
            "docstring_lines": docstring_lines,
            "code_lines": code_lines,
            "function_metrics": function_metrics,
            "type_hint_pct": type_hint_pct,
        }

    except Exception as e:
        print(f"Error analyzing file {file_path}: {e}")
        return None


def get_dir_name(directory, dir_names=None):
    """Get directory display name from path or provided names."""
    if dir_names and len(dir_names) > 0:
        return dir_names.pop(0)
    return Path(directory).parts[-1]  # Get final directory in path


def process_directories(directories, dir_names=None):
    """Process all Python files in the given directories and track metrics per directory."""
    directory_metrics = {}

    total_files = 0
    processed_files = 0

    # Create a list of directory names if not provided
    if not dir_names:
        dir_names = []
    else:
        # Make a copy to avoid modifying the original
        dir_names = dir_names.copy()

    for directory in directories:
        dir_name = get_dir_name(directory, dir_names)

        # Initialize metrics for this directory
        directory_metrics[dir_name] = {
            "function_loc": [],
            "comment_ratio": [],
            "docstring_ratio": [],
            "var_lengths": [],
            "cyclomatic_complexity": [],
            "cognitive_complexity": [],
            "max_nested_level": [],
            "return_count": [],
            "type_hint_pct": [],
        }

        directory_path = Path(directory)
        dir_files = 0
        dir_processed = 0

        for py_file in directory_path.glob("**/*.py"):
            dir_files += 1
            total_files += 1
            print(f"Processing {py_file}")

            metrics = analyze_file(py_file)
            if not metrics:
                continue

            dir_processed += 1
            processed_files += 1

            # Store type hint percentage
            directory_metrics[dir_name]["type_hint_pct"].append(metrics["type_hint_pct"])

            # Compute file-level metrics
            if metrics["code_lines"] > 0:
                comment_ratio = metrics["comment_lines"] / metrics["code_lines"]
                docstring_ratio = metrics["docstring_lines"] / metrics["code_lines"]

                directory_metrics[dir_name]["comment_ratio"].append(comment_ratio)
                directory_metrics[dir_name]["docstring_ratio"].append(docstring_ratio)

            # Collect function-level metrics
            for func in metrics["function_metrics"]:
                directory_metrics[dir_name]["function_loc"].append(func["loc"])
                directory_metrics[dir_name]["var_lengths"].extend(func["var_lengths"])
                directory_metrics[dir_name]["cyclomatic_complexity"].append(
                    func["cyclomatic_complexity"]
                )
                directory_metrics[dir_name]["cognitive_complexity"].append(
                    func["cognitive_complexity"]
                )
                directory_metrics[dir_name]["max_nested_level"].append(func["max_nested_level"])
                directory_metrics[dir_name]["return_count"].append(func["return_count"])

        print(f"Processed {dir_processed} of {dir_files} Python files in {dir_name}.")

    print(f"Processed {processed_files} of {total_files} total Python files.")
    return directory_metrics


def plot_histogram(data_dict, metric_name, x_label, output_path, bins=None, cap_value=None):
    """Create and save a histogram plot for a metric across directories."""
    plt.figure()

    # Create a colormap for directories
    num_dirs = len(data_dict)
    colors = plt.cm.tab10(np.linspace(0, 1, num_dirs))

    # Calculate max value for binning if needed
    max_value = 0
    for dir_name, metrics in data_dict.items():
        if metrics[metric_name]:
            if cap_value:
                max_value = max(max_value, min(max(metrics[metric_name]), cap_value))
            else:
                max_value = max(max_value, max(metrics[metric_name]))

    # Create bins if not provided
    if bins is None:
        if metric_name in ["max_nested_level", "return_count"]:
            # For small integer values, use unit bins
            bins = np.arange(0, max_value + 2, 1)
        elif metric_name in ["cyclomatic_complexity", "cognitive_complexity"]:
            # For complexity metrics
            bins = np.arange(0, max_value + 5, 1)
        elif metric_name == "type_hint_pct":
            # For percentages
            bins = np.linspace(0, 100, 21)
        else:
            # Default binning
            bins = 20

    if max_value > 0:
        for i, (dir_name, metrics) in enumerate(data_dict.items()):
            if metrics[metric_name]:
                values = metrics[metric_name]
                if cap_value:
                    # Cap values for better visualization
                    values = [min(v, cap_value) for v in values]

                plt.hist(
                    values,
                    bins=bins,
                    alpha=0.6,
                    color=colors[i],
                    edgecolor="black",
                    linewidth=1.0,
                    density=True,
                    label=dir_name,
                    histtype="step",
                    fill=True,
                )

        plt.xlabel(x_label)
        plt.ylabel("Density")
        title = f"Distribution of {x_label}"
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return True
    return False


def create_histogram_plots(directory_metrics, output_dir):
    """Create and save histogram plots for the collected metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Set publication-ready style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 24,
            "xtick.labelsize": 14,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "figure.figsize": (10, 6),
            "figure.dpi": 300,
        }
    )

    # Define metrics to plot
    metrics_to_plot = [
        (
            "function_loc",
            "Lines of Code per Function (lower is better)",
            "function_loc_histogram.png",
            100,
        ),
        (
            "comment_ratio",
            "Comment to Code Ratio (higher is better)",
            "comment_ratio_histogram.png",
            1.0,
        ),
        (
            "docstring_ratio",
            "Docstring to Code Ratio (higher is better)",
            "docstring_ratio_histogram.png",
            1.0,
        ),
        (
            "var_lengths",
            "Variable Name Length (characters) (higher is better)",
            "var_length_histogram.png",
            30,
        ),
        (
            "cyclomatic_complexity",
            "Cyclomatic Complexity (branch points + 1) (lower is better)",
            "cyclomatic_complexity_histogram.png",
            20,
        ),
        (
            "cognitive_complexity",
            "Cognitive Complexity (lower is better)",
            "cognitive_complexity_histogram.png",
            30,
        ),
        (
            "max_nested_level",
            "Nested Block Depth (lower is better)",
            "nested_depth_histogram.png",
            10,
        ),
        (
            "return_count",
            "Return Statements per Function (closer to 1 is better)",
            "return_count_histogram.png",
            10,
        ),
        (
            "type_hint_pct",
            "Type Hint Usage (%) (higher is better)",
            "type_hint_pct_histogram.png",
            None,
        ),
    ]

    # Generate all plots
    plots_created = []
    for metric, label, filename, cap in metrics_to_plot:
        if plot_histogram(directory_metrics, metric, label, output_path / filename, cap_value=cap):
            plots_created.append(metric)

    print(f"Created plots for: {', '.join(plots_created)}")
    print(f"Plots saved to {output_path}")


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze Python code metrics across directories")
    parser.add_argument(
        "directories", nargs="+", help="Directories containing Python files to analyze"
    )
    parser.add_argument("--output", "-o", default="./plots", help="Output directory for plots")
    parser.add_argument(
        "--names", "-n", nargs="*", help="Optional display names for directories (in order)"
    )

    args = parser.parse_args()

    print(f"Analyzing Python files in directories: {args.directories}")

    if args.names and len(args.names) < len(args.directories):
        print(
            f"Warning: Fewer names ({len(args.names)}) provided than directories ({len(args.directories)}). "
            f"Using default names for remaining directories."
        )

    directory_metrics = process_directories(args.directories, args.names)

    # Print summary statistics
    print("\nSummary Statistics by Directory:")
    for dir_name, metrics in directory_metrics.items():
        print(f"\n{dir_name}:")
        print(f"  Total functions analyzed: {len(metrics['function_loc'])}")
        if metrics["function_loc"]:
            print(f"  Average function size: {np.mean(metrics['function_loc']):.2f} lines")
        if metrics["comment_ratio"]:
            print(f"  Average comment to code ratio: {np.mean(metrics['comment_ratio']):.2f}")
        if metrics["docstring_ratio"]:
            print(f"  Average docstring to code ratio: {np.mean(metrics['docstring_ratio']):.2f}")
        if metrics["var_lengths"]:
            print(
                f"  Average variable name length: {np.mean(metrics['var_lengths']):.2f} characters"
            )
        if metrics["cyclomatic_complexity"]:
            print(
                f"  Average cyclomatic complexity: {np.mean(metrics['cyclomatic_complexity']):.2f}"
            )
        if metrics["cognitive_complexity"]:
            print(f"  Average cognitive complexity: {np.mean(metrics['cognitive_complexity']):.2f}")
        if metrics["max_nested_level"]:
            print(f"  Average max nesting depth: {np.mean(metrics['max_nested_level']):.2f}")
        if metrics["return_count"]:
            print(
                f"  Average return statements per function: {np.mean(metrics['return_count']):.2f}"
            )
        if metrics["type_hint_pct"]:
            print(f"  Average type hint usage: {np.mean(metrics['type_hint_pct']):.2f}%")

    create_histogram_plots(directory_metrics, args.output)


if __name__ == "__main__":
    main()
