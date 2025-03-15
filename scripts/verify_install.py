#!/usr/bin/env python3
"""
Eidosian Installation Verification Protocol

This script performs a comprehensive verification of the Glyph Forge
installation to ensure all components are correctly installed and
operational with zero compromise.
"""
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple


# Terminal styling constants
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def verify_module(module_name: str) -> bool:
    """Verify a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def run_verification() -> Tuple[bool, Dict[str, List[str]]]:
    """Run comprehensive installation verification."""
    results = {
        "success": [],
        "failure": []
    }
    
    # Core dependencies
    core_deps = ["numpy", "PIL", "glyph_forge"]
    for dep in core_deps:
        if verify_module(dep):
            results["success"].append(f"Core dependency: {dep}")
        else:
            results["failure"].append(f"Core dependency: {dep}")
    
    # Test dependencies for development
    test_deps = ["pytest", "pytest_mock", "pytest_cov"]
    for dep in test_deps:
        if verify_module(dep):
            results["success"].append(f"Test dependency: {dep}")
        else:
            results["failure"].append(f"Test dependency: {dep}")
    
    # Verify specific Glyph Forge modules are importable
    forge_modules = [
        "glyph_forge.services.image_to_Glyph",
        "glyph_forge.utils.alphabet_manager",
    ]
    for module in forge_modules:
        if verify_module(module):
            results["success"].append(f"Glyph Forge module: {module}")
        else:
            results["failure"].append(f"Glyph Forge module: {module}")
    
    return len(results["failure"]) == 0, results


def main() -> int:
    """Main entry point for verification script."""
    print(f"\n{BOLD}{BLUE}⚡ GLYPH FORGE: INSTALLATION VERIFICATION ⚡{RESET}\n")
    
    # Run verification
    success, results = run_verification()
    
    # Print results
    for item in results["success"]:
        print(f"{GREEN}✓ {item}{RESET}")
    
    for item in results["failure"]:
        print(f"{RED}✗ {item}{RESET}")
    
    # Print summary
    print("\n" + "="*60)
    if success:
        print(f"{BOLD}{GREEN}✅ VERIFICATION COMPLETE - INSTALLATION VALIDATED{RESET}")
    else:
        print(f"{BOLD}{RED}❌ VERIFICATION FAILED - FIX INSTALLATION ISSUES{RESET}")
    print("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())