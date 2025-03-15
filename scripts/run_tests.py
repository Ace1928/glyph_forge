#!/usr/bin/env python3
"""
⚡ GLYPH FORGE TEST EXECUTOR ⚡

Quantum-precise test runner for the Glyph Forge ecosystem.
Executes tests with surgical precision, maximum parallelism,
and zero-compromise reporting.

Usage:
    python scripts/run_tests.py [options]

Options:
    --verbose       Enable detailed output
    --coverage      Generate coverage report
    --output FILE   Save results to specified file
    --focus PATTERN Focus on specific test pattern
    --skip PATTERN  Skip tests matching pattern
    --quick         Run only critical path tests
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
import shutil
from typing import List, Dict, Any, Optional, Tuple
import xml.etree.ElementTree as ET
import json


# Terminal styling for maximum clarity
class Style:
    """Terminal styling with atomic precision."""
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    CLEAR_LINE = "\033[K"
    
    @staticmethod
    def apply(text: str, *styles: str) -> str:
        """Apply multiple styles with automatic reset."""
        return "".join(styles) + text + Style.RESET


# Test execution constants
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
COVERAGE_DIR = PROJECT_ROOT / "coverage"
PYTEST_CACHE = PROJECT_ROOT / ".pytest_cache"


class TestResult:
    """Container for atomic test results with zero ambiguity."""
    def __init__(self):
        self.total: int = 0
        self.passed: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.errors: int = 0
        self.execution_time: float = 0.0
        self.coverage: Optional[float] = None
        self.failing_tests: List[Dict[str, Any]] = []


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with zero redundancy."""
    parser = argparse.ArgumentParser(
        description="Glyph Forge Test Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\nOptions:")[0]
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                      help='Generate coverage report')
    parser.add_argument('--output', '-o', type=str,
                      help='Save results to specified file')
    parser.add_argument('--focus', '-f', type=str,
                      help='Focus on specific test pattern')
    parser.add_argument('--skip', '-s', type=str,
                      help='Skip tests matching pattern')
    parser.add_argument('--quick', '-q', action='store_true',
                      help='Run only critical path tests')
    parser.add_argument('--junit', '-j', action='store_true',
                      help='Generate JUnit XML report')
    parser.add_argument('--html', action='store_true',
                      help='Generate HTML report')
    
    return parser.parse_args()


def ensure_test_environment() -> bool:
    """
    Ensure the test environment is correctly configured.
    
    Returns:
        True if environment is ready, False otherwise
    """
    # Check if pytest is installed
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except subprocess.CalledProcessError:
        print(Style.apply(
            "Error: pytest not installed. Run: pip install pytest pytest-cov",
            Style.RED, Style.BOLD
        ))
        return False
    
    # Check if src directory exists
    if not SRC_DIR.is_dir():
        print(Style.apply(
            f"Error: Source directory not found at {SRC_DIR}",
            Style.RED, Style.BOLD
        ))
        return False
    
    # Check if tests directory exists
    if not TESTS_DIR.is_dir():
        print(Style.apply(
            f"Error: Tests directory not found at {TESTS_DIR}",
            Style.RED, Style.BOLD
        ))
        return False
    
    return True


def build_pytest_command(args: argparse.Namespace) -> List[str]:
    """
    Build pytest command with precision-optimized parameters.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        List of command arguments
    """
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=src/glyph_forge",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage/coverage.xml"
        ])
        if args.html:
            cmd.append("--cov-report=html:coverage/html")
    
    # Add JUnit report
    if args.junit:
        os.makedirs("reports", exist_ok=True)
        cmd.append("--junitxml=reports/junit.xml")
    
    # Add HTML report
    if args.html and not args.coverage:
        os.makedirs("reports", exist_ok=True)
        cmd.append("--html=reports/report.html")
    
    # Add test focus pattern
    if args.focus:
        cmd.append(f"-k {args.focus}")
    
    # Add test skip pattern
    if args.skip:
        cmd.append(f"-k 'not {args.skip}'")
    
    # Quick mode - only run critical tests
    if args.quick:
        cmd.append("-m critical")
    
    # Always add tests directory
    cmd.append(str(TESTS_DIR))
    
    return cmd


def execute_tests(cmd: List[str]) -> Tuple[int, str, str]:
    """
    Execute test command with maximum efficiency.
    
    Args:
        cmd: Command to execute
        
    Returns:
        Tuple of (return code, stdout, stderr)
    """
    print(Style.apply("\n▶ Executing tests", Style.BOLD, Style.CYAN))
    
    # Run the command and capture output
    process = subprocess.Popen(
        " ".join(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True
    )
    
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def parse_test_results(stdout: str, stderr: str, junit_path: Optional[str] = None) -> TestResult:
    """
    Parse test results from output with atomic precision.
    
    Args:
        stdout: Standard output from test execution
        stderr: Standard error from test execution
        junit_path: Path to JUnit XML report if available
        
    Returns:
        TestResult object
    """
    result = TestResult()
    
    # Try parsing from JUnit XML if available
    if junit_path and os.path.exists(junit_path):
        try:
            tree = ET.parse(junit_path)
            root = tree.getroot()
            testsuite = root.find('testsuite')
            if testsuite is not None:
                result.total = int(testsuite.get('tests', '0'))
                result.errors = int(testsuite.get('errors', '0'))
                result.failed = int(testsuite.get('failures', '0'))
                result.skipped = int(testsuite.get('skipped', '0'))
                result.passed = result.total - result.failed - result.errors - result.skipped
                result.execution_time = float(testsuite.get('time', '0'))
                
                # Collect failing tests
                for testcase in testsuite.findall('.//testcase'):
                    failure = testcase.find('failure')
                    error = testcase.find('error')
                    if failure is not None or error is not None:
                        result.failing_tests.append({
                            'name': testcase.get('name', 'Unknown'),
                            'classname': testcase.get('classname', 'Unknown'),
                            'time': float(testcase.get('time', '0')),
                            'message': (failure.get('message') if failure is not None else 
                                       error.get('message') if error is not None else 'Unknown')
                        })
                return result
        except Exception as e:
            print(f"Warning: Could not parse JUnit XML report: {e}")
    
    # Fall back to parsing stdout/stderr
    for line in stdout.splitlines():
        # Look for the summary line: "4 passed, 1 skipped, 2 failed"
        if " passed" in line and ("failed" in line or "skipped" in line or "error" in line):
            # Extract numbers using regex
            import re
            passed_match = re.search(r'(\d+) passed', line)
            failed_match = re.search(r'(\d+) failed', line)
            skipped_match = re.search(r'(\d+) skipped', line)
            error_match = re.search(r'(\d+) error', line)
            
            if passed_match:
                result.passed = int(passed_match.group(1))
            if failed_match:
                result.failed = int(failed_match.group(1))
            if skipped_match:
                result.skipped = int(skipped_match.group(1))
            if error_match:
                result.errors = int(error_match.group(1))
                
            result.total = result.passed + result.failed + result.skipped + result.errors
            break
    
    # Extract coverage information if available
    coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', stdout)
    if coverage_match:
        result.coverage = float(coverage_match.group(1))
    
    # Extract execution time
    time_match = re.search(r'in (\d+\.\d+)s', stdout)
    if time_match:
        result.execution_time = float(time_match.group(1))
    
    return result


def display_results(result: TestResult) -> None:
    """Display test results with maximum clarity."""
    print("\n" + "=" * 50)
    print(Style.apply(" GLYPH FORGE TEST RESULTS ", Style.BOLD, Style.CYAN))
    print("=" * 50)
    
    # Summary
    print(f"\nTotal tests: {result.total}")
    print(f"Passed: {Style.apply(str(result.passed), Style.GREEN, Style.BOLD)}")
    
    if result.failed > 0:
        print(f"Failed: {Style.apply(str(result.failed), Style.RED, Style.BOLD)}")
    else:
        print(f"Failed: {result.failed}")
        
    if result.errors > 0:
        print(f"Errors: {Style.apply(str(result.errors), Style.RED, Style.BOLD)}")
    else:
        print(f"Errors: {result.errors}")
        
    print(f"Skipped: {Style.apply(str(result.skipped), Style.YELLOW)}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    # Coverage
    if result.coverage is not None:
        coverage_color = (
            Style.RED if result.coverage < 70 else 
            Style.YELLOW if result.coverage < 90 else 
            Style.GREEN
        )
        print(f"Coverage: {Style.apply(f'{result.coverage:.1f}%', coverage_color, Style.BOLD)}")
    
    # Failing tests details
    if result.failing_tests:
        print("\n" + Style.apply("Failing Tests:", Style.BOLD, Style.RED))
        for i, test in enumerate(result.failing_tests, 1):
            print(f"{i}. {test['classname']}.{test['name']}")
            print(f"   - {test['message']}")
    
    # Final status
    print("\n" + "=" * 50)
    if result.failed > 0 or result.errors > 0:
        print(Style.apply(" ❌ TESTS FAILED ", Style.RED, Style.BOLD))
    else:
        print(Style.apply(" ✅ TESTS PASSED ", Style.GREEN, Style.BOLD))
    print("=" * 50 + "\n")


def save_results(result: TestResult, output_path: str) -> None:
    """Save test results to file with zero data loss."""
    try:
        # Create parent directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert to JSON-serializable dict
        result_dict = {
            "total": result.total,
            "passed": result.passed,
            "failed": result.failed,
            "skipped": result.skipped,
            "errors": result.errors,
            "execution_time": result.execution_time,
            "coverage": result.coverage,
            "failing_tests": result.failing_tests
        }
        
        # Write with atomic replace strategy
        temp_path = output_path + '.tmp'
        with open(temp_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        # Rename for atomic replacement
        if os.path.exists(output_path):
            os.replace(temp_path, output_path)
        else:
            os.rename(temp_path, output_path)
            
        print(Style.apply(f"Results saved to {output_path}", Style.CYAN))
    except Exception as e:
        print(Style.apply(f"Error saving results: {str(e)}", Style.RED))


def main() -> int:
    """Main entry point with zero-friction flow."""
    args = parse_arguments()
    
    # Print banner
    print(Style.apply("\n⚡ GLYPH FORGE TEST EXECUTOR ⚡\n", Style.BOLD, Style.CYAN))
    
    # Check environment
    if not ensure_test_environment():
        return 1
    
    # Prepare directories
    os.makedirs(COVERAGE_DIR, exist_ok=True)
    
    # Build and execute test command
    start_time = time.time()
    cmd = build_pytest_command(args)
    
    print(Style.apply("Command:", Style.BOLD))
    print(" ".join(cmd))
    print()
    
    exit_code, stdout, stderr = execute_tests(cmd)
    
    # Parse and display results
    junit_path = "reports/junit.xml" if args.junit and os.path.exists("reports/junit.xml") else None
    result = parse_test_results(stdout, stderr, junit_path)
    display_results(result)
    
    # Save results if requested
    if args.output:
        save_results(result, args.output)
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f}s\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())