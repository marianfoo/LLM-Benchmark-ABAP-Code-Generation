from datetime import datetime, timezone
from html import unescape
import json
import logging
import os
import time
import traceback
from typing import Dict, List, Literal, Optional, Tuple, TypedDict
from abap_adt_py.adt_client import AdtClient
import xml.etree.ElementTree as ET
import re
import tqdm

# Optional abaplint integration (imported lazily to avoid dependency issues)
try:
    from abaplint_check import run_abaplint, format_lint_feedback, is_abaplint_available
    ABAPLINT_AVAILABLE = is_abaplint_available()
except ImportError:
    ABAPLINT_AVAILABLE = False
    run_abaplint = None
    format_lint_feedback = None


# =============================================================================
# Retry Utility for flaky SAP ADT connections
# =============================================================================

def _is_transient_error(exc: Exception) -> bool:
    """Check if the exception looks like a transient SAP server error (500/503)
    or an auth/session error (401/403) that can be fixed by re-login."""
    msg = str(exc)
    return any(code in msg for code in [
        "500 -", "503 -", "500 Internal", "503 Service",
        "401 -", "403 -",
    ])


# Reference to the current AdtClient, set by run_abap_interaction so the
# retry helper can re-login on session errors without creating new sessions.
_current_client: Optional[AdtClient] = None


def _retry_adt(fn, *args, max_retries: int = 3, base_delay: float = 5.0, **kwargs):
    """
    Retry an ADT call with exponential back-off on transient server errors.
    On auth errors (401/403), attempts a single re-login before retrying.
    Raises the original exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            result = fn(*args, **kwargs)
            # Throttle after every successful call to avoid overwhelming
            # a trial SAP system (prevents EG Memory exhaustion / session pileup)
            time.sleep(ADT_THROTTLE_DELAY)
            return result
        except Exception as e:
            last_exc = e
            if attempt < max_retries and _is_transient_error(e):
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                # On auth errors, try re-login (reuses existing requests.Session)
                msg = str(e)
                if any(code in msg for code in ["401 -", "403 -"]) and _current_client:
                    try:
                        _current_client.login()
                    except Exception:
                        pass  # will retry the whole call anyway
            else:
                raise
    raise last_exc  # type: ignore


# Throttle delay (seconds) between ADT calls.
# Main fix was removing per-rep login() which caused 80+ sessions / EG Memory exhaustion.
# This is just a small safety buffer. Set to 0 for fast dedicated systems.
ADT_THROTTLE_DELAY = 0.05

# Delay (seconds) between repetitions within a prompt
REP_DELAY = 0

# Delay (seconds) between prompts
PROMPT_DELAY = 0

# Prefix used in feedback messages for transient/infra errors so retry mode
# can identify them and re-attempt the SAP test.
INFRA_FEEDBACK_PREFIX = "[INFRA] Transient ADT error"


def log_failure(
    model_name: str,
    prompt_key: str,
    rep_idx: int,
    round_num: int,
    attempt: int,
    stage: str,
    transient: bool,
    message: str,
):
    """Append one line to the model's failure log (data/<model>_abap_test_failures.log)."""
    log_path = f"data/{model_name.replace(':', '_')}_abap_test_failures.log"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Keep message on a single line (truncate to 300 chars)
    short_msg = message.replace("\n", " ")[:300]
    line = (
        f"{ts}\t{model_name}\t{prompt_key}\trep={rep_idx}\tround={round_num}"
        f"\tattempt={attempt}\tstage={stage}\ttransient={transient}\t{short_msg}\n"
    )
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass  # Don't let logging failures break the run


# =============================================================================
# Tiered Result Tracking
# =============================================================================

class TierResult(TypedDict, total=False):
    """Result tracking for each validation tier."""
    lint_pass: Optional[bool]           # abaplint preflight (None if not run)
    sap_syntax_pass: Optional[bool]     # SAP syntax check passed
    sap_activation_pass: Optional[bool] # SAP activation passed
    unit_pass: Optional[bool]           # ABAP Unit tests passed
    error_stage: Optional[str]          # First stage that failed: "canonicalization" | "syntax" | "activation" | "unit"
    error_message: Optional[str]        # Error message from the failing stage
    canonicalization_report: Optional[Dict]  # Details about what was renamed


def create_tier_result() -> TierResult:
    """Create an empty tier result structure."""
    return {
        "lint_pass": None,
        "sap_syntax_pass": None,
        "sap_activation_pass": None,
        "unit_pass": None,
        "error_stage": None,
        "error_message": None,
        "canonicalization_report": None,
    }


def load_tiers(model_name: str) -> Dict[str, List[Dict[str, TierResult]]]:
    """Load tier results from file, or return empty structure."""
    filename = f"data/{model_name.replace(':', '_')}_tiers.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def save_tiers(model_name: str, tiers: Dict[str, List[Dict[str, TierResult]]]):
    """Save tier results to file."""
    filename = f"data/{model_name.replace(':', '_')}_tiers.json"
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(tiers, file, indent=4, ensure_ascii=False)


def get_round_number(chat: List[Dict]) -> int:
    """
    Determine the current round number based on the conversation.
    Round 0 = initial response, Round 1+ = after feedback.
    """
    # Count assistant messages to determine round
    assistant_count = sum(1 for msg in chat if msg.get("role") == "assistant")
    return assistant_count - 1  # 0-indexed: first response is round 0


# =============================================================================
# Run Manifest Generation
# =============================================================================

def get_git_sha() -> Optional[str]:
    """Get the current git commit SHA."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def get_abaplint_version() -> Optional[str]:
    """Get the installed abaplint version."""
    import subprocess
    try:
        # Try local install first
        abaplint_cmd = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "tools", "node_modules", ".bin", "abaplint"
        )
        if not os.path.exists(abaplint_cmd):
            abaplint_cmd = "npx"
            args = [abaplint_cmd, "@abaplint/cli", "--version"]
        else:
            args = [abaplint_cmd, "--version"]
        
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def create_run_manifest(
    model_name: str,
    repetitions: int = 10,
    max_rounds: int = 5,
    use_canonicalization: bool = True,
    use_abaplint: bool = True,
) -> Dict:
    """
    Create a run manifest with metadata for reproducibility.
    
    Returns:
        Dictionary containing run metadata
    """
    from datetime import datetime
    
    timestamp = datetime.now()
    run_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{model_name.replace(':', '_').replace('/', '_')}"
    
    manifest = {
        "run_id": run_id,
        "model": model_name,
        "timestamp": timestamp.isoformat(),
        "git_sha": get_git_sha(),
        "config": {
            "repetitions": repetitions,
            "max_rounds": max_rounds,
            "use_canonicalization": use_canonicalization,
            "use_abaplint": use_abaplint,
        },
        "versions": {
            "abaplint": get_abaplint_version(),
            "python": None,  # Can be filled in if needed
        }
    }
    
    return manifest


def save_run_manifest(manifest: Dict) -> str:
    """
    Save a run manifest to the runs directory.
    
    Args:
        manifest: Run manifest dictionary
    
    Returns:
        Path to the saved manifest file
    """
    run_id = manifest["run_id"]
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs", run_id)
    os.makedirs(runs_dir, exist_ok=True)
    
    manifest_path = os.path.join(runs_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest_path


def load_run_manifest(run_id: str) -> Optional[Dict]:
    """Load a run manifest by run ID."""
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "runs", run_id, "manifest.json"
    )
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def list_runs() -> List[str]:
    """List all available run IDs."""
    runs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")
    if not os.path.exists(runs_dir):
        return []
    return sorted([
        d for d in os.listdir(runs_dir) 
        if os.path.isdir(os.path.join(runs_dir, d))
    ])


def compute_tier_statistics(model_name: str) -> Dict:
    """
    Compute aggregate statistics from tier results.
    
    Returns statistics for:
    - Direct (Round 0) pass rates
    - With feedback (best across all rounds) pass rates
    - Failure stage distribution
    """
    tiers = load_tiers(model_name)
    
    stats = {
        "total_tasks": 0,
        "total_repetitions": 0,
        "direct": {
            "syntax_pass": 0,
            "activation_pass": 0,
            "unit_pass": 0,
        },
        "with_feedback": {
            "syntax_pass": 0,
            "activation_pass": 0,
            "unit_pass": 0,
        },
        "failure_stages": {
            "canonicalization": 0,
            "create": 0,
            "syntax": 0,
            "activation": 0,
            "unit": 0,
        },
        "success_round_distribution": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
    }
    
    for prompt_file, repetitions in tiers.items():
        stats["total_tasks"] += 1
        
        for rep_idx, rounds in enumerate(repetitions):
            if not rounds:
                continue
            stats["total_repetitions"] += 1
            
            # Direct (Round 0) results
            round_0 = rounds.get("round_0", {})
            if round_0.get("sap_syntax_pass"):
                stats["direct"]["syntax_pass"] += 1
            if round_0.get("sap_activation_pass"):
                stats["direct"]["activation_pass"] += 1
            if round_0.get("unit_pass"):
                stats["direct"]["unit_pass"] += 1
            
            # With feedback (best across all rounds)
            best_syntax = False
            best_activation = False
            best_unit = False
            success_round = None
            final_error_stage = None
            
            for round_key in sorted(rounds.keys()):
                round_data = rounds[round_key]
                if round_data.get("sap_syntax_pass"):
                    best_syntax = True
                if round_data.get("sap_activation_pass"):
                    best_activation = True
                if round_data.get("unit_pass"):
                    best_unit = True
                    if success_round is None:
                        round_num = int(round_key.split("_")[1])
                        success_round = round_num
                # Track final error stage (from last round)
                if round_data.get("error_stage"):
                    final_error_stage = round_data["error_stage"]
            
            if best_syntax:
                stats["with_feedback"]["syntax_pass"] += 1
            if best_activation:
                stats["with_feedback"]["activation_pass"] += 1
            if best_unit:
                stats["with_feedback"]["unit_pass"] += 1
            
            if success_round is not None:
                stats["success_round_distribution"][success_round] += 1
            elif final_error_stage:
                if final_error_stage not in stats["failure_stages"]:
                    stats["failure_stages"][final_error_stage] = 0
                stats["failure_stages"][final_error_stage] += 1
    
    # Compute percentages
    total_reps = stats["total_repetitions"] or 1  # Avoid division by zero
    stats["direct"]["syntax_pass_rate"] = stats["direct"]["syntax_pass"] / total_reps
    stats["direct"]["activation_pass_rate"] = stats["direct"]["activation_pass"] / total_reps
    stats["direct"]["unit_pass_rate"] = stats["direct"]["unit_pass"] / total_reps
    stats["with_feedback"]["syntax_pass_rate"] = stats["with_feedback"]["syntax_pass"] / total_reps
    stats["with_feedback"]["activation_pass_rate"] = stats["with_feedback"]["activation_pass"] / total_reps
    stats["with_feedback"]["unit_pass_rate"] = stats["with_feedback"]["unit_pass"] / total_reps
    
    return stats


# =============================================================================
# Contract Extraction and Canonicalization
# =============================================================================

def extract_contract(prompt_file: str) -> Dict[str, str]:
    """
    Extract expected class and method names from the unit test file.
    
    Args:
        prompt_file: The prompt filename (e.g., "1.txt" or "erp_000.txt")
    
    Returns:
        Dictionary with 'class_name' and 'method_name' keys
    """
    # Handle both regular tasks (1.txt) and ERP tasks (erp_000.txt)
    task_id = prompt_file[:-4]  # Remove .txt extension
    
    if task_id.startswith("erp_"):
        # ERP task: erp_000.txt -> z_humaneval_erp_000
        class_name = f"z_humaneval_{task_id}"
    else:
        # Regular task: 1.txt -> z_humaneval_001
        class_name = f"z_humaneval_{task_id.zfill(3)}"
    
    unittest_file = f"dataset/abap_unittests/{class_name}_test.abap"
    
    with open(unittest_file, "r", encoding="utf-8") as file:
        test_src = file.read()
    
    # Find the method call pattern: z_humaneval_XXX=>method_name(
    pattern = rf"{re.escape(class_name)}=>(\w+)\s*\("
    match = re.search(pattern, test_src, re.IGNORECASE)
    
    if match:
        method_name = match.group(1)
    else:
        # Fallback: couldn't extract method name
        method_name = None
    
    return {
        "class_name": class_name,
        "method_name": method_name,
        "unittest_file": unittest_file,
    }


def canonicalize_code(src: str, contract: Dict[str, str]) -> Tuple[str, Dict]:
    """
    Rename class and method in LLM output to match the expected contract.
    
    This preserves all other code (types, helper methods, data declarations)
    and only renames the class and the main public method.
    
    Args:
        src: The ABAP source code from the LLM
        contract: Dictionary with 'class_name' and 'method_name' from extract_contract()
    
    Returns:
        Tuple of (canonicalized_source, report_dict)
        report_dict contains details about what was changed
    """
    report = {
        "original_class": None,
        "original_method": None,
        "target_class": contract["class_name"],
        "target_method": contract["method_name"],
        "class_renamed": False,
        "method_renamed": False,
        "warnings": [],
    }
    
    target_class = contract["class_name"].upper()
    target_method = contract["method_name"].upper() if contract["method_name"] else None
    
    # Step 1: Find the original class name
    # Pattern matches: CLASS classname DEFINITION
    class_def_pattern = r"CLASS\s+(\w+)\s+DEFINITION"
    class_match = re.search(class_def_pattern, src, re.IGNORECASE)
    
    if not class_match:
        report["warnings"].append("Could not find CLASS ... DEFINITION in source")
        return src, report
    
    original_class = class_match.group(1)
    report["original_class"] = original_class
    
    # Step 2: Replace class name throughout the source (case-insensitive)
    if original_class.upper() != target_class:
        # Replace all occurrences of the class name (preserving case structure)
        src = re.sub(
            rf"\b{re.escape(original_class)}\b",
            target_class,
            src,
            flags=re.IGNORECASE
        )
        report["class_renamed"] = True
    
    # Step 3: Find and rename the main method (if target_method is specified)
    if target_method:
        # Find CLASS-METHODS or METHODS declaration for the public method
        # Pattern: CLASS-METHODS methodname or METHODS methodname
        # We look for the first public method declaration
        method_pattern = r"(?:CLASS-METHODS|METHODS)\s*:?\s*(\w+)"
        
        # Find all method declarations in PUBLIC SECTION
        public_section_match = re.search(
            r"PUBLIC\s+SECTION\.(.*?)(?:PROTECTED\s+SECTION|PRIVATE\s+SECTION|ENDCLASS)",
            src,
            re.IGNORECASE | re.DOTALL
        )
        
        if public_section_match:
            public_section = public_section_match.group(1)
            method_matches = re.findall(method_pattern, public_section, re.IGNORECASE)
            
            if method_matches:
                # Take the first public method as the main method
                original_method = method_matches[0]
                report["original_method"] = original_method
                
                if original_method.upper() != target_method:
                    # Replace all occurrences of the method name
                    src = re.sub(
                        rf"\b{re.escape(original_method)}\b",
                        target_method,
                        src,
                        flags=re.IGNORECASE
                    )
                    report["method_renamed"] = True
            else:
                report["warnings"].append("Could not find method declaration in PUBLIC SECTION")
        else:
            report["warnings"].append("Could not find PUBLIC SECTION in source")
    
    return src, report


def get_unittest_src(unittest_file: str):
    """Read and return the unit test source code."""
    with open(unittest_file, "r") as file:
        test_src = file.read()
        return test_src


def unittest_find_method_calls(
    default_class_name: str, unittest_file: str
) -> List[str]:
    """Find method calls in unit test file (legacy function for backwards compatibility)."""
    test_src = get_unittest_src(unittest_file)
    finds = list(set(re.findall(f"{default_class_name}=>.*\\(", test_src)))
    return finds


def class_find_method_calls(client: AdtClient, class_name: str):
    class_uri = f"/sap/bc/adt/oo/classes/{class_name}"
    resp = client.object_structure(class_uri)
    methods = [
        m
        for m in resp["components"]
        if "type" in m
        and m["type"] == "CLAS/OM"
        and "visibility" in m
        and m["visibility"] == "public"
    ]
    method_names = [m["name"] for m in methods if "name" in m]
    return method_names


def build_unittest_src(client: AdtClient, class_name: str, prompt_file: str, chat):
    default_class_name = f"z_humaneval_{str.zfill(prompt_file[:-4], 3)}"
    unittest_file = f"dataset/abap_unittests/{default_class_name}_test.abap"

    unittest_method_calls = unittest_find_method_calls(
        default_class_name, unittest_file
    )
    class_methods = class_find_method_calls(client, class_name)

    if len(unittest_method_calls) == 1 and len(class_methods) == 1:
        unittest_src = get_unittest_src(unittest_file)
        new_method_call = f"{class_name}=>{class_methods[0]}("
        unittest_src = unittest_src.replace(unittest_method_calls[0], new_method_call)
        # print(f"replaced {unittest_method_calls[0]} with {new_method_call}")
        return unittest_src
    else:
        add_to_chat(chat, "There should only be the one public method.")
        return ""


def run_unit_tests(client: AdtClient, class_uri: str, chat):
    result = _retry_adt(client.run_unit_test, class_uri)

    if len(result) > 0:
        add_to_chat(
            chat,
            f"The unit test failed with the following errors:\n{result}\n Try to fix the class, not the unit test.",
        )
        success = False
    else:
        add_to_chat(
            chat,
            f"The unit tests were successful.",
        )
        success = True
    return success


def create_test_class_include(client: AdtClient, class_name: str, class_uri: str):
    try:
        lock_handle: str = _retry_adt(client.lock, class_uri)
        _retry_adt(client.create_test_class_include, class_name, lock_handle)
    finally:
        _retry_adt(client.unlock, class_uri, lock_handle)


def syntax_check(client: AdtClient, class_uri: str, src: str, chat) -> bool:
    syntaxcheck_results = _retry_adt(client.syntax_check, class_uri, class_uri, src)
    if len(syntaxcheck_results) > 0:
        add_to_chat(
            chat,
            f"The syntax check failed with the following errors:\n{syntaxcheck_results}",
        )
        success = False
    else:
        success = True
    return success


def syntax_check_unittest(client: AdtClient, class_uri: str, src: str, chat) -> bool:
    syntaxcheck_results = _retry_adt(client.syntax_check, class_uri, class_uri, src)
    if len(syntaxcheck_results) > 0:
        add_to_chat(
            chat,
            f"The unittest syntax check failed with the following errors:\n{syntaxcheck_results}\nTry to fix the class, not the unit test.",
        )
        success = False
    else:
        success = True
    return success


def create_class(client: AdtClient, class_name: str, chat) -> bool:
    try:
        _retry_adt(
            client.create,
            object_type="CLAS/OC",
            name=class_name,
            description="Automatically created class for benchmarking purposes",
            parent="$TMP",
        )
        success = True
    except Exception as e:
        error_text = str(e)
        # Try to extract the body after the first line (status line)
        try:
            error_text = error_text[error_text.index("\n") + 1 :].strip()
            xml_error = parse_error_xml(error_text)
        except Exception:
            # Fallback: use the raw exception text
            xml_error = {"exception_type": "UNKNOWN", "message": str(e)[:500], "longtext": ""}
        add_to_chat(
            chat,
            f"The class could not be created due to the following error:\n{xml_error}",
        )
        success = False
    return success


def set_source(
    client: AdtClient,
    class_uri: str,
    src: str,
    chat,
    suffix: Literal["/source/main", "/includes/testclasses"] = "/source/main",
):
    try:
        lock_handle: str = _retry_adt(client.lock, class_uri)
        try:
            _retry_adt(client.set_object_source, f"{class_uri}{suffix}", src, lock_handle)
            success = True
        finally:
            _retry_adt(client.unlock, class_uri, lock_handle)
    except Exception as e:
        error_text = str(e)
        try:
            error_text = error_text[error_text.index("\n") + 1 :]
            xml_error = parse_error_xml(error_text)
        except Exception:
            xml_error = {"exception_type": "UNKNOWN", "message": str(e)[:500], "longtext": ""}
        add_to_chat(
            chat,
            f"The source code could not be set due to the following error:\n{xml_error}",
        )
        success = False
    return success


def activate_class(client: AdtClient, class_uri: str, chat) -> bool:
    try:
        _retry_adt(client.activate, class_uri, class_uri)
        success = True
    except Exception as e:
        error_text = str(e)
        add_to_chat(
            chat,
            f"The activation failed with the following error:\n{error_text}",
        )
        success = False
    return success


def delete_class(client: AdtClient, class_uri: str):
    lock_handle: str = _retry_adt(client.lock, class_uri)
    try:
        _retry_adt(client.delete, class_uri, lock_handle)
    finally:
        _retry_adt(client.unlock, class_uri, lock_handle)


def clean_longtext(longtext_html: str) -> str:
    clean_text = ""
    if longtext_html:
        decoded_html = unescape(longtext_html)
        try:
            text = re.sub(r"<br\s*/?>", "\n", decoded_html, flags=re.IGNORECASE)
            text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
            text = re.sub(r"<p[^>]*>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<[^>]+>", "", text)
            lines = [line.strip() for line in text.split("\n")]
            cleaned_lines = []
            prev_empty = False
            for line in lines:
                if line:
                    cleaned_lines.append(line)
                    prev_empty = False
                elif not prev_empty:
                    cleaned_lines.append("")
                    prev_empty = True
            clean_text = "\n".join(cleaned_lines).strip()
        except:
            clean_text = decoded_html
    return clean_text


def parse_error_xml(xml_string):
    """Parse an ADT XML error response. Falls back gracefully for non-XML
    responses (e.g. HTML 500 error pages returned by ICM)."""
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        # Not valid XML (e.g. HTML error page) – return a best-effort dict
        # Strip HTML tags to extract a readable message
        plain = re.sub(r"<[^>]+>", " ", xml_string)
        plain = re.sub(r"\s+", " ", plain).strip()
        return {
            "exception_type": "PARSE_ERROR",
            "message": plain[:500] if plain else xml_string[:500],
            "longtext": "",
        }

    exception_type = (
        root.find(".//type").get("id") if root.find(".//type") is not None else None,  # type: ignore
    )[0]
    message = (
        root.find(".//message").text if root.find(".//message") is not None else None,  # type: ignore
    )[0]

    properties = {}
    for entry in root.findall(".//entry"):
        key = entry.get("key")
        value = entry.text
        properties[key] = value

    longtext_html = properties.get("LONGTEXT", "")
    clean_text = clean_longtext(longtext_html)

    return {
        "exception_type": exception_type,
        "message": message,
        "longtext": clean_text,
    }


def add_to_chat(chat, content):
    chat.append(
        {
            "role": "user",
            "content": content,
        }
    )


def run_single_prompt(
    client,
    prompt_file: str,
    chats: list,
    use_canonicalization: bool = True,
    use_abaplint: bool = True,
    abaplint_hard_gate: bool = False,
    model_name: str = "",
):
    """
    Process all repetitions for a single prompt against the SAP system.
    
    This is the inner loop extracted from run_abap_interaction() so it can be
    called from both the sequential runner and the parallel runner.

    Args:
        client: An already-logged-in AdtClient instance.
        prompt_file: The prompt key (e.g. "erp_000.txt").
        chats: List of chat histories (one per repetition) for this prompt.
        use_canonicalization: If True, canonicalize LLM output to match fixed tests.
        use_abaplint: If True, run abaplint preflight check before SAP.
        abaplint_hard_gate: If True, skip SAP if abaplint has parse errors.
        model_name: Model identifier used for failure logging.

    Returns:
        tuple: (updated_chats, tiers_for_prompt)
            - updated_chats: The chats list with any feedback messages appended.
            - tiers_for_prompt: A list of dicts, one per repetition, with tier results.
    """
    # Extract the contract (expected class/method names) for this task
    contract = extract_contract(prompt_file)

    # Initialize tier storage for this prompt
    tiers_for_prompt = [{} for _ in range(len(chats))]

    for rep_idx, chat in enumerate(tqdm.tqdm(chats, desc=f"Processing {prompt_file}", leave=False)):

        if chat[-1]["role"] == "user":
            continue
        if chat[-1]["content"] == "The unit tests were successful.":
            continue

        # Initialize tier result for this attempt
        tier = create_tier_result()
        round_num = get_round_number(chat)
        class_uri = None  # Initialize for cleanup

        try:
            skip_remaining_steps = False
            last_message = chat[-1]
            src = last_message["content"]

            if use_canonicalization:
                # New approach: Canonicalize LLM output to match fixed unit test contract
                src, canon_report = canonicalize_code(src, contract)
                tier["canonicalization_report"] = canon_report

                if canon_report["warnings"]:
                    # If canonicalization had warnings, report them
                    tier["error_stage"] = "canonicalization"
                    tier["error_message"] = "; ".join(canon_report["warnings"])
                    add_to_chat(
                        chat,
                        f"Code structure issue: {tier['error_message']}. "
                        f"Please provide a valid ABAP class with a public static method.",
                    )
                    skip_remaining_steps = True
                else:
                    # Use the canonical class name from the contract
                    class_name = contract["class_name"].upper()
                    class_uri = f"/sap/bc/adt/oo/classes/{class_name}"
            else:
                # Legacy approach: Extract class name from LLM output
                match_class_name = re.match(
                    r"CLASS\s+(\w+)", src, re.MULTILINE | re.IGNORECASE
                )
                if match_class_name:
                    class_name = match_class_name.group(1)
                    class_uri = f"/sap/bc/adt/oo/classes/{class_name}"
                else:
                    tier["error_stage"] = "canonicalization"
                    tier["error_message"] = "Class name not found in source"
                    add_to_chat(
                        chat,
                        f"Class name not found.",
                    )
                    skip_remaining_steps = True

            # Run abaplint preflight check (if enabled and available)
            if not skip_remaining_steps and use_abaplint and ABAPLINT_AVAILABLE and run_abaplint:
                lint_result = run_abaplint(src)
                tier["lint_pass"] = lint_result["success"]

                if lint_result["parse_error"] and abaplint_hard_gate:
                    # Hard gate: Skip SAP if code cannot be parsed
                    tier["error_stage"] = "lint"
                    tier["error_message"] = format_lint_feedback(lint_result)
                    add_to_chat(chat, tier["error_message"])
                    skip_remaining_steps = True
                elif not lint_result["success"]:
                    # Soft gate: Record lint failure but proceed to SAP
                    # (SAP is the authoritative judge)
                    pass  # Continue to SAP validation

            if not skip_remaining_steps:
                success = create_class(client, class_name, chat)
                if not success:
                    tier["error_stage"] = "create"
                    tier["error_message"] = chat[-1]["content"] if chat else "Class creation failed"
                    skip_remaining_steps = True

            if not skip_remaining_steps:
                success = syntax_check(client, class_uri, src, chat)
                tier["sap_syntax_pass"] = success
                if not success:
                    tier["error_stage"] = "syntax"
                    tier["error_message"] = chat[-1]["content"] if chat else "Syntax check failed"
                    skip_remaining_steps = True

            if not skip_remaining_steps:
                success = set_source(client, class_uri, src, chat)
                if not success:
                    tier["error_stage"] = "syntax"
                    tier["error_message"] = chat[-1]["content"] if chat else "Set source failed"
                    skip_remaining_steps = True

            if not skip_remaining_steps:
                success = activate_class(client, class_uri, chat)
                tier["sap_activation_pass"] = success
                if not success:
                    tier["error_stage"] = "activation"
                    tier["error_message"] = chat[-1]["content"] if chat else "Activation failed"
                    skip_remaining_steps = True

            if not skip_remaining_steps:
                create_test_class_include(client, class_name, class_uri)

                if use_canonicalization:
                    # New approach: Use the fixed unit test directly
                    unittest_src = get_unittest_src(contract["unittest_file"])
                else:
                    # Legacy approach: Dynamically adapt unit test to call model's method
                    unittest_src = build_unittest_src(client, class_name, prompt_file, chat)
                    if unittest_src == "":
                        skip_remaining_steps = True

                if not skip_remaining_steps:
                    success = syntax_check_unittest(
                        client, class_uri + "/includes/testclasses", unittest_src, chat
                    )
                    if not success:
                        # Unit test syntax errors are usually due to signature mismatch
                        tier["error_stage"] = "unit"
                        tier["error_message"] = chat[-1]["content"] if chat else "Unit test syntax check failed"
                        skip_remaining_steps = True

                if not skip_remaining_steps:
                    success = set_source(
                        client, class_uri, unittest_src, chat, "/includes/testclasses"
                    )
                    if not success:
                        tier["error_stage"] = "unit"
                        tier["error_message"] = chat[-1]["content"] if chat else "Set test source failed"
                        skip_remaining_steps = True

                if not skip_remaining_steps:
                    success = activate_class(client, class_uri, chat)
                    if not success:
                        tier["error_stage"] = "activation"
                        tier["error_message"] = chat[-1]["content"] if chat else "Test activation failed"
                        skip_remaining_steps = True

                if not skip_remaining_steps:
                    success = run_unit_tests(client, class_uri, chat)
                    tier["unit_pass"] = success
                    if not success:
                        tier["error_stage"] = "unit"
                        tier["error_message"] = chat[-1]["content"] if chat else "Unit tests failed"

        except Exception as exc:
            # ------------------------------------------------------------------
            # Catch-all: record transient/infra failures so they can be retried.
            # Without this, an unhandled exception aborts the entire run and
            # leaves many conversations with NO feedback at all.
            # ------------------------------------------------------------------
            exc_short = str(exc)[:300]
            tier["error_stage"] = "infra"
            tier["error_message"] = exc_short

            # Only append feedback if the chat doesn't already end with a user
            # message (avoids double-appending on re-raises inside _retry_adt).
            if chat[-1]["role"] != "user":
                add_to_chat(
                    chat,
                    f"{INFRA_FEEDBACK_PREFIX}: {exc_short}",
                )

            # Log the failure for later analysis / retry
            log_failure(
                model_name=model_name,
                prompt_key=prompt_file,
                rep_idx=rep_idx,
                round_num=round_num,
                attempt=1,
                stage="infra",
                transient=True,
                message=exc_short,
            )
            tqdm.tqdm.write(
                f"  [INFRA] {prompt_file} rep={rep_idx}: {exc_short[:120]}"
            )

        finally:
            # Cleanup: Always try to delete the class
            if class_uri:
                try:
                    delete_class(client, class_uri)
                except Exception:
                    pass

            # Store tier result for this round
            tiers_for_prompt[rep_idx][f"round_{round_num}"] = tier

        # Cooldown between repetitions to avoid overwhelming trial SAP systems
        time.sleep(REP_DELAY)

    return chats, tiers_for_prompt


def run_abap_interaction(
    model_name: str, 
    use_canonicalization: bool = True,
    use_abaplint: bool = True,
    abaplint_hard_gate: bool = False,
):
    """
    Run ABAP interaction for all prompts of a given model.
    
    Args:
        model_name: Name of the model (used for finding the data file)
        use_canonicalization: If True, canonicalize LLM output to match fixed tests.
                             If False, use legacy dynamic test adaptation.
        use_abaplint: If True, run abaplint preflight check before SAP.
        abaplint_hard_gate: If True, skip SAP if abaplint has parse errors.
                           If False (default), always proceed to SAP (soft gate).
    """
    global _current_client
    client: AdtClient = AdtClient(
        sap_host="http://localhost:50000",
        username="DEVELOPER",
        password="ABAPtr2023#00",
        client="001",
        language="EN",
    )
    client.login()
    # Store reference so _retry_adt can re-login on auth errors
    # without creating new sessions (prevents SM04 session pileup)
    _current_client = client

    filename = f"data/{model_name.replace(':', '_')}.json"
    with open(filename, "r", encoding="utf-8") as file:
        prompt_files = json.load(file)
    
    # Load existing tier results or initialize empty structure
    all_tiers = load_tiers(model_name)
    
    for prompt_file, chats in tqdm.tqdm(
        prompt_files.items(), "Processing prompts", leave=False
    ):
        updated_chats, prompt_tiers = run_single_prompt(
            client, prompt_file, chats,
            use_canonicalization=use_canonicalization,
            use_abaplint=use_abaplint,
            abaplint_hard_gate=abaplint_hard_gate,
            model_name=model_name,
        )

        # Write updated chats back into prompt_files (for incremental save)
        prompt_files[prompt_file] = updated_chats

        # Merge prompt tier results
        all_tiers[prompt_file] = prompt_tiers

        # Save after each prompt (incremental save)
        with open(filename, "w", encoding="utf-8") as file:
            file.write(json.dumps(prompt_files, indent=4, ensure_ascii=False))
        save_tiers(model_name, all_tiers)

        # Cooldown between prompts
        time.sleep(PROMPT_DELAY)