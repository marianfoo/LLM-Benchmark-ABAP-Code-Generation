"""
Abaplint integration for local preflight checking.

This module provides fast local linting using abaplint before
sending code to SAP. Used as a soft gate (diagnostic tier) and
for providing faster feedback to the LLM.
"""

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, TypedDict


class LintError(TypedDict):
    """Structure for a single lint error."""
    severity: str       # "Error" | "Warning" | "Information"
    message: str        # Error message text
    line: Optional[int] # Line number (1-indexed)
    column: Optional[int]  # Column number


class LintResult(TypedDict):
    """Result of running abaplint on code."""
    success: bool           # True if no errors (warnings OK)
    parse_error: bool       # True if abaplint couldn't parse the code at all
    errors: List[LintError] # List of error details
    error_count: int        # Number of errors
    warning_count: int      # Number of warnings
    raw_output: Optional[str]  # Raw abaplint output for debugging


# Path to abaplint - prefer local install in tools/ directory
ABAPLINT_LOCAL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    "tools", 
    "node_modules", 
    ".bin", 
    "abaplint"
)


def get_abaplint_command() -> List[str]:
    """
    Get the command to run abaplint.
    Prefers local install, falls back to npx.
    """
    if os.path.exists(ABAPLINT_LOCAL):
        return [ABAPLINT_LOCAL]
    else:
        # Fall back to npx (slower but works without local install)
        return ["npx", "@abaplint/cli"]


def run_abaplint(code: str, timeout: int = 30) -> LintResult:
    """
    Run abaplint on ABAP code and return structured results.
    
    Args:
        code: ABAP source code to lint
        timeout: Timeout in seconds for abaplint execution
    
    Returns:
        LintResult with success status and error details
    """
    result: LintResult = {
        "success": False,
        "parse_error": False,
        "errors": [],
        "error_count": 0,
        "warning_count": 0,
        "raw_output": None,
    }
    
    # Write code to a temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", 
        suffix=".abap", 
        delete=False,
        encoding="utf-8"
    ) as tmp_file:
        tmp_file.write(code)
        tmp_path = tmp_file.name
    
    try:
        # Build the abaplint command
        cmd = get_abaplint_command() + [
            "--format", "json",
            tmp_path
        ]
        
        # Run abaplint
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(__file__))  # Run from project root
        )
        
        result["raw_output"] = proc.stdout or proc.stderr
        
        # Parse JSON output
        if proc.stdout:
            try:
                lint_output = json.loads(proc.stdout)
                
                # abaplint JSON format: list of issues per file
                # Each issue has: severity, message, start (line, column)
                for file_issues in lint_output:
                    if isinstance(file_issues, dict) and "issues" in file_issues:
                        for issue in file_issues["issues"]:
                            error: LintError = {
                                "severity": issue.get("severity", "Error"),
                                "message": issue.get("message", "Unknown error"),
                                "line": issue.get("start", {}).get("row"),
                                "column": issue.get("start", {}).get("col"),
                            }
                            result["errors"].append(error)
                            
                            if error["severity"] == "Error":
                                result["error_count"] += 1
                            elif error["severity"] == "Warning":
                                result["warning_count"] += 1
                
                # Success if no errors (warnings are OK)
                result["success"] = result["error_count"] == 0
                
                # Check for parse errors specifically
                for error in result["errors"]:
                    if "parser" in error["message"].lower() or "unexpected" in error["message"].lower():
                        result["parse_error"] = True
                        break
                        
            except json.JSONDecodeError:
                # JSON parse failed - treat as parse error
                result["parse_error"] = True
                result["errors"].append({
                    "severity": "Error",
                    "message": f"abaplint output not valid JSON: {proc.stdout[:500]}",
                    "line": None,
                    "column": None,
                })
        else:
            # No stdout - check stderr for errors
            if proc.stderr:
                result["parse_error"] = True
                result["errors"].append({
                    "severity": "Error",
                    "message": f"abaplint error: {proc.stderr[:500]}",
                    "line": None,
                    "column": None,
                })
            elif proc.returncode == 0:
                # No output and success return code = no issues
                result["success"] = True
                
    except subprocess.TimeoutExpired:
        result["errors"].append({
            "severity": "Error",
            "message": f"abaplint timed out after {timeout} seconds",
            "line": None,
            "column": None,
        })
    except FileNotFoundError:
        result["errors"].append({
            "severity": "Error",
            "message": "abaplint not found. Install with: npm install @abaplint/cli",
            "line": None,
            "column": None,
        })
    except Exception as e:
        result["errors"].append({
            "severity": "Error",
            "message": f"abaplint execution failed: {str(e)}",
            "line": None,
            "column": None,
        })
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return result


def format_lint_feedback(lint_result: LintResult, max_errors: int = 5) -> str:
    """
    Format lint errors into a feedback message for the LLM.
    
    Args:
        lint_result: Result from run_abaplint()
        max_errors: Maximum number of errors to include (to avoid huge prompts)
    
    Returns:
        Formatted error message string
    """
    if lint_result["success"]:
        return ""
    
    lines = ["ABAP syntax check found the following issues:"]
    
    shown = 0
    for error in lint_result["errors"]:
        if shown >= max_errors:
            remaining = len(lint_result["errors"]) - shown
            lines.append(f"... and {remaining} more issue(s)")
            break
        
        loc = ""
        if error["line"]:
            loc = f" (line {error['line']}"
            if error["column"]:
                loc += f", column {error['column']}"
            loc += ")"
        
        lines.append(f"- {error['severity']}{loc}: {error['message']}")
        shown += 1
    
    return "\n".join(lines)


def is_abaplint_available() -> bool:
    """Check if abaplint is available on this system."""
    try:
        cmd = get_abaplint_command() + ["--version"]
        proc = subprocess.run(cmd, capture_output=True, timeout=10)
        return proc.returncode == 0
    except:
        return False
