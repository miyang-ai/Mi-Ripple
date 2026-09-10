"""Deterministic diagnosis-guided MIRAGE restoration pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

from .common import load_rgb, now_iso, sha256_of, write_json
from .diagnosis import diag_board, diagnose
from .notch import notch_image
from .reference import PARAMS as REFERENCE_PARAMS
from .reference import detect_faces, reference_grade_clean
from .regeneration import (
    DEFAULT_MODEL,
    RegenClient,
    client_from_env,
    regenerate,
)
from .spatial import iso_clean
from .verify import verify, verify_board

ACTIONS = (
    "diagnose",
    "reference_clean",
    "regenerate",
    "iso_clean",
    "notch",
    "verify",
    "finish",
    "request_human",
)


@dataclass
class State:
    source: Path
    out_dir: Path
    stem: str
    current: Path
    allow_regen: bool
    max_regen: int
    regen_model: str
    quality: str
    phase: str = "triage"
    diag: dict | None = None
    diag_path: Path | None = None
    regen_count: int = 0
    reference: Path | None = None
    iso_applied: bool = False
    notch_applied: bool = False
    verify_result: dict | None = None
    steps: list[dict] = field(default_factory=list)
    finished: bool = False
    outcome: str = ""
    human_note: str = ""
    authorize_regen: Callable[[], bool] | None = None


def structured_scales(flags: dict) -> bool:
    return bool(
        flags.get("granule_pervasive") or flags.get("scale_structured")
    )


def allowed_actions(state: State) -> list[str]:
    if state.finished:
        return []
    if state.diag is None:
        return ["diagnose"]
    flags = state.diag["flags"]
    if state.phase == "triage":
        if structured_scales(flags):
            if state.allow_regen and state.regen_count < state.max_regen:
                return ["reference_clean", "request_human"]
            return ["request_human", "iso_clean"]
        if flags["granule_flat"] or flags["granule_suspected"]:
            return ["iso_clean", "notch", "finish"]
        if flags["lattice"]:
            return ["notch", "finish"]
        return ["finish"]
    if state.phase == "reference_ready":
        return ["regenerate", "request_human"]
    if state.phase == "deliverable":
        actions = []
        if not state.iso_applied and flags["granule_flat"]:
            actions.append("iso_clean")
        if not state.notch_applied:
            actions.append("notch")
        actions.append(
            "verify"
            if (state.iso_applied or state.notch_applied)
            and state.verify_result is None
            else "finish"
        )
        return actions
    return ["finish"]


def decide_rules(state: State) -> tuple[str, str]:
    actions = allowed_actions(state)
    if not actions:
        return "finish", "No further action is available."
    flags = state.diag["flags"] if state.diag else {}
    if state.diag is None:
        return "diagnose", "Diagnosis is required before treatment."
    if state.phase == "triage":
        if structured_scales(flags):
            evidence = []
            if flags["granule_pervasive"]:
                evidence.append(
                    f"{state.diag['granule']['flagged_windows']} "
                    "flat windows indicate pervasive granules"
                )
            if flags.get("scale_structured"):
                scale = state.diag["scale_index"]
                evidence.append(
                    f"whole-frame scale index {scale['scale_tile_pct']}% "
                    f"({scale['level']})"
                )
            evidence_text = "; ".join(evidence)
            if "reference_clean" in actions:
                if (
                    state.authorize_regen is not None
                    and not state.authorize_regen()
                ):
                    return (
                        "request_human",
                        f"{evidence_text}; regeneration was not authorized.",
                    )
                return (
                    "reference_clean",
                    f"{evidence_text}; content-entangled texture requires "
                    "cleaned-reference regeneration.",
                )
            if state.regen_count:
                return (
                    "request_human",
                    f"After {state.regen_count} regeneration round(s), "
                    f"{evidence_text}; the retry limit was reached.",
                )
            return (
                "request_human",
                f"{evidence_text}; regeneration was not enabled.",
            )
        if flags["granule_flat"]:
            return (
                "iso_clean",
                "Granules were confirmed in flat regions; use strict masked reduction.",
            )
        if flags["granule_suspected"]:
            state.human_note = (
                "One suspected granule window requires visual inspection; "
                "automatic spatial reduction was skipped."
            )
            return (
                "notch",
                "Granules are only suspected; apply the low-distortion notch route.",
            )
        if flags["lattice"]:
            return (
                "notch",
                "An isolated periodic lattice was detected.",
            )
        return "finish", "No supported artifact class was detected."
    if state.phase == "reference_ready":
        return (
            "regenerate",
            "Regenerate once using only the cleaned reference.",
        )
    if state.phase == "deliverable":
        reasons = {
            "iso_clean": "Reduce confirmed flat-region granules.",
            "notch": "Attenuate isolated periodic components.",
            "verify": "Verify aligned filtering distortion.",
            "finish": "No further processing is required.",
        }
        return actions[0], reasons[actions[0]]
    return "finish", "Processing is complete."


def observation_xml(state: State) -> str:
    diagnosis = state.diag or {}
    parts = [
        (
            f'<observation phase="{state.phase}" '
            f'regen_count="{state.regen_count}" '
            f'allow_regen="{str(state.allow_regen).lower()}">'
        )
    ]
    if diagnosis:
        lattice = diagnosis["lattice"]
        granule = diagnosis["granule"]
        parts.append(
            
                f'<lattice detected="{str(lattice["detected"]).lower()}" '
                f'max_peak_excess="{lattice["max_peak_excess"]}" '
                f'isolated_peaks="{lattice["isolated_components_kept"]}"/>'
            
        )
        parts.append(
            
                f'<granule level="{granule["level"]}" '
                f'flagged="{granule["flagged_windows"]}" '
                f'flat_windows="{len(granule["flat_windows"])}"/>'
            
        )
        scale = diagnosis.get("scale_index")
        if scale:
            parts.append(
                
                    f'<scale_index pct="{scale["scale_tile_pct"]}" '
                    f'level="{scale["level"]}" '
                    f'flagged_tiles="{scale["flagged_tiles"]}" '
                    f'tiles="{scale["tiles"]}"/>'
                
            )
    if state.verify_result:
        verification = state.verify_result
        reasons = "".join(
            f"<reason>{escape(reason)}</reason>"
            for reason in verification["reasons"]
        )
        parts.append(
            
                f'<verify passed="{str(verification["passed"]).lower()}">'
                f"{reasons}</verify>"
            
        )
    actions = "".join(
        f"<action>{action}</action>" for action in allowed_actions(state)
    )
    parts.append(f"<allowed_actions>{actions}</allowed_actions>")
    parts.append("</observation>")
    return "".join(parts)


def _save(state: State, array: np.ndarray, suffix: str) -> Path:
    path = state.out_dir / f"{state.stem}_{suffix}.png"
    Image.fromarray(array).save(path)
    return path


def act(
    state: State,
    action: str,
    reason: str,
    client: RegenClient | None,
) -> dict:
    result: dict = {"action": action}
    if action == "diagnose":
        rgb = load_rgb(state.current)
        tag = f"regen{state.regen_count}" if state.regen_count else "input"
        diagnosis = diagnose(
            rgb,
            scale_heat_out=(
                state.out_dir / f"{state.stem}_{tag}_scaleheat.png"
            ),
        )
        diagnosis.update(
            {
                "stage": "diagnose",
                "source": str(state.current),
                "source_sha256": sha256_of(state.current),
                "time": now_iso(),
            }
        )
        diagnosis["board"] = str(
            diag_board(
                rgb,
                diagnosis,
                state.out_dir / f"{state.stem}_{tag}_diag_board.png",
                f"Diagnosis: {state.current.name}",
            )
        )
        state.diag_path = write_json(
            state.out_dir / f"{state.stem}_{tag}_diag.json",
            diagnosis,
        )
        state.diag = diagnosis
        result.update(
            {
                "lattice": diagnosis["flags"]["lattice"],
                "granule_level": diagnosis["granule"]["level"],
                "scale_tile_pct": diagnosis["scale_index"][
                    "scale_tile_pct"
                ],
                "scale_level": diagnosis["scale_index"]["level"],
                "board": diagnosis["board"],
                "scale_heatmap": diagnosis["scale_index"].get("heatmap"),
            }
        )
        if (
            state.phase == "triage"
            and state.regen_count
            and not structured_scales(diagnosis["flags"])
        ):
            state.phase = "deliverable"
    elif action == "reference_clean":
        rgb = load_rgb(state.current)
        boxes, detection_method = detect_faces(rgb)
        output, _, stats = reference_grade_clean(
            rgb,
            REFERENCE_PARAMS,
            boxes,
        )
        state.reference = _save(
            state,
            output,
            f"regen{state.regen_count + 1}_refclean",
        )
        write_json(
            state.reference.with_suffix(".json"),
            {
                "stage": "reference_clean",
                "grade": "reference_only",
                "not_a_deliverable": True,
                "source": str(state.current),
                "source_sha256": sha256_of(state.current),
                "face_detection": detection_method,
                **stats,
                "time": now_iso(),
            },
        )
        state.phase = "reference_ready"
        result.update(
            {
                "reference": str(state.reference),
                "face_detection": detection_method,
                "face_boxes": stats["face_boxes"],
            }
        )
    elif action == "regenerate":
        if client is None or state.reference is None:
            raise RuntimeError(
                "Regeneration requires a client and cleaned reference."
            )
        state.regen_count += 1
        output = state.out_dir / (
            f"{state.stem}_regen{state.regen_count}.png"
        )
        sidecar = regenerate(
            state.reference,
            output,
            client,
            state.regen_model,
            state.quality,
            extra_sidecar={
                "regen_round": state.regen_count,
                "decision_reason": reason,
            },
        )
        state.current = output
        state.diag = None
        state.phase = "triage"
        result.update(
            {
                "output": str(output),
                "saved_size": sidecar["saved_size"],
                "size_verified": sidecar["size_verified"],
            }
        )
    elif action == "iso_clean":
        previous = state.current
        output, weight, stats = iso_clean(load_rgb(previous), 1.0, "strict")
        state.current = _save(state, output, "iso_strict")
        Image.fromarray((weight * 255).astype(np.uint8)).save(
            state.out_dir / f"{state.stem}_iso_strict_mask.png"
        )
        write_json(
            state.current.with_suffix(".json"),
            {
                "stage": "iso_clean",
                "grade": "deliverable",
                "source_before": str(previous),
                **stats,
                "time": now_iso(),
            },
        )
        state.iso_applied = True
        state.phase = "deliverable"
        result.update({"output": str(state.current), **stats["mask"]})
    elif action == "notch":
        previous = state.current
        output, stats = notch_image(load_rgb(previous))
        suffix = "iso_strict_notch" if state.iso_applied else "notch"
        state.current = _save(state, output, suffix)
        write_json(
            state.current.with_suffix(".json"),
            {
                "stage": "notch",
                "grade": "deliverable",
                "source_before": str(previous),
                **stats,
                "time": now_iso(),
            },
        )
        state.notch_applied = True
        state.phase = "deliverable"
        result.update(
            {
                "output": str(state.current),
                **{
                    key: value
                    for key, value in stats["per_channel"]["L"].items()
                    if key != "peaks_top"
                },
            }
        )
    elif action == "verify":
        before_path = (
            state.source
            if state.regen_count == 0
            else state.out_dir / f"{state.stem}_regen{state.regen_count}.png"
        )
        before = load_rgb(before_path)
        after = load_rgb(state.current)
        flat_boxes = [
            row["box"]
            for row in state.diag["granule"]["flat_windows"]
        ]
        structure_boxes = [
            row["box"]
            for row in state.diag["granule"]["oriented_windows"]
        ] + [list(box) for box in detect_faces(before)[0]]
        verification = verify(
            before,
            after,
            flat_boxes,
            structure_boxes,
        )
        verification.update(
            {
                "stage": "verify",
                "before": str(before_path),
                "after": str(state.current),
                "time": now_iso(),
            }
        )
        verification["board"] = str(
            verify_board(
                before,
                after,
                verification,
                state.out_dir / f"{state.stem}_verify_board.png",
                f"Verification: {state.current.name}",
            )
        )
        write_json(
            state.out_dir / f"{state.stem}_verify.json",
            verification,
        )
        state.verify_result = verification
        result.update(
            {
                "passed": verification["passed"],
                "reasons": verification["reasons"],
                "board": verification["board"],
            }
        )
        if not verification["passed"] and state.iso_applied:
            rollback_output, stats = notch_image(before)
            state.current = _save(
                state,
                rollback_output,
                "notch_only_rollback",
            )
            write_json(
                state.current.with_suffix(".json"),
                {
                    "stage": "notch",
                    "grade": "deliverable",
                    "rollback_from": "iso_strict_notch",
                    "rollback_reasons": verification["reasons"],
                    **stats,
                    "time": now_iso(),
                },
            )
            rollback_verification = verify(
                before,
                load_rgb(state.current),
                flat_boxes,
                structure_boxes,
            )
            rollback_verification.update(
                {
                    "stage": "verify",
                    "before": str(before_path),
                    "after": str(state.current),
                    "rollback": True,
                    "time": now_iso(),
                }
            )
            rollback_verification["board"] = str(
                verify_board(
                    before,
                    load_rgb(state.current),
                    rollback_verification,
                    state.out_dir
                    / f"{state.stem}_verify_rollback_board.png",
                    f"Rollback verification: {state.current.name}",
                )
            )
            write_json(
                state.out_dir / f"{state.stem}_verify_rollback.json",
                rollback_verification,
            )
            state.verify_result = rollback_verification
            state.iso_applied = False
            result.update(
                {
                    "rollback": True,
                    "rollback_passed": rollback_verification["passed"],
                }
            )
    elif action == "request_human":
        state.finished = True
        state.outcome = "needs_human_decision"
        state.human_note = reason
    elif action == "finish":
        state.finished = True
        state.outcome = (
            "delivered"
            if state.verify_result is None or state.verify_result["passed"]
            else "delivered_with_verify_failure"
        )
    else:
        raise ValueError(f"Unknown action: {action}")
    return result


def _emitter(
    on_event: Callable[[dict], None] | None,
) -> Callable[[dict], None]:
    if on_event is None:
        return lambda _event: None
    warned = False

    def emit(event: dict) -> None:
        nonlocal warned
        try:
            on_event(event)
        except Exception as error:
            if not warned:
                warned = True
                print(
                    f"[warning] progress receiver failed: {error!r}",
                    file=sys.stderr,
                )

    return emit


def run(
    source: Path,
    out_dir: Path,
    allow_regen: bool = False,
    max_regen: int = 2,
    regen_model: str = DEFAULT_MODEL,
    quality: str = "high",
    client: RegenClient | None = None,
    max_steps: int = 14,
    on_event: Callable[[dict], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    authorize_regen: Callable[[], bool] | None = None,
) -> dict:
    """Run diagnosis, treatment, and verification for one image."""
    source = Path(source)
    out_dir = Path(out_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    emit = _emitter(on_event)
    state = State(
        source=source,
        out_dir=out_dir,
        stem=source.stem,
        current=source,
        allow_regen=allow_regen,
        max_regen=max_regen,
        regen_model=regen_model,
        quality=quality,
        authorize_regen=authorize_regen,
    )
    if allow_regen and client is None:
        client = client_from_env()
    trace = [
        "<trace>",
        (
            f'<meta source="{escape(str(source))}" '
            f'sha256="{sha256_of(source)}" '
            f'started="{now_iso()}" mode="rules"/>'
        ),
    ]
    cancelled = False
    for step_number in range(1, max_steps + 1):
        if state.finished:
            break
        if should_cancel is not None and should_cancel():
            cancelled = True
            state.finished = True
            state.outcome = "cancelled"
            state.human_note = (
                f"Cancelled after {step_number - 1} completed step(s)."
            )
            break
        observation = observation_xml(state)
        action, reason = decide_rules(state)
        step = {
            "n": step_number,
            "current": str(state.current),
            "action": action,
            "by": "rules",
            "reason": reason,
        }
        emit({"type": "step_start", "step": dict(step)})
        try:
            result = act(state, action, reason, client)
        except Exception as error:
            result = {"action": action, "error": repr(error)}
            state.finished = True
            state.outcome = "failed"
            state.human_note = repr(error)
        step["result"] = result
        state.steps.append(step)
        emit({"type": "step_end", "step": dict(step)})
        trace.append(
            
                f'<step n="{step_number}">{observation}'
                f'<decision action="{action}" by="rules">'
                f"{escape(reason)}</decision>"
                f"<result>{escape(json.dumps(result, ensure_ascii=False))}"
                "</result></step>"
            
        )
        print(f"[{step_number}] {action:16s} {reason[:90]}")
    if not state.finished:
        state.finished = True
        state.outcome = "step_limit"
        state.human_note = f"Pipeline did not finish within {max_steps} steps."
    final = out_dir / f"{state.stem}_restored.png"
    if state.outcome.startswith("delivered"):
        shutil.copyfile(state.current, final)
    if cancelled:
        print(f"[-] cancelled {state.human_note}")
    summary = {
        "stage": "restore",
        "outcome": state.outcome,
        "human_note": state.human_note,
        "source": str(source),
        "source_sha256": sha256_of(source),
        "final": str(final) if final.exists() else None,
        "final_sha256": sha256_of(final) if final.exists() else None,
        "final_is_source_copy": final.exists() and state.current == source,
        "regen_rounds": state.regen_count,
        "mode": "rules",
        "diag": str(state.diag_path) if state.diag_path else None,
        "verify": (
            {
                "passed": state.verify_result["passed"],
                "reasons": state.verify_result["reasons"],
                "board": state.verify_result.get("board"),
            }
            if state.verify_result
            else None
        ),
        "steps": state.steps,
        "acceptance": (
            "A person must inspect the boards and decide whether to accept "
            "the candidate."
        ),
        "time": now_iso(),
    }
    trace.append(f"<outcome>{escape(state.outcome)}</outcome>")
    trace.append("</trace>")
    (out_dir / f"{state.stem}_trace.xml").write_text(
        "\n".join(trace) + "\n",
        encoding="utf-8",
    )
    write_json(out_dir / f"{state.stem}_restored.json", summary)
    emit(
        {
            "type": "outcome",
            "outcome": state.outcome,
            "human_note": state.human_note,
            "final": summary["final"],
            "regen_rounds": state.regen_count,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose and restore digital-ripple artifacts."
    )
    parser.add_argument("image")
    parser.add_argument("out_dir")
    parser.add_argument(
        "--allow-regen",
        action="store_true",
        help="Permit bounded regeneration when filtering is unsafe.",
    )
    parser.add_argument("--max-regen", type=int, default=2)
    parser.add_argument("--regen-model", default=DEFAULT_MODEL)
    parser.add_argument("--quality", default="high")
    arguments = parser.parse_args()
    summary = run(
        Path(arguments.image),
        Path(arguments.out_dir),
        arguments.allow_regen,
        arguments.max_regen,
        arguments.regen_model,
        arguments.quality,
    )
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "outcome",
                    "human_note",
                    "final",
                    "regen_rounds",
                    "mode",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
