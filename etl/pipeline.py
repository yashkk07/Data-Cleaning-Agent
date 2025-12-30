import logging
from typing import Dict, Any
import pandas as pd

from etl.validate.validator import sanitize_feedback
from etl.extract.reader import read_structured_safe
from etl.profile.profiler import profile_dataframe
from etl.profile.serializer import ensure_json_serializable
from etl.llm.planner import generate_plan
from etl.executor.tool_executor import execute_plan
from etl.validate.validator import validate_transformation
from etl.assessment.confidence import compute_confidence
from etl.assessment.readiness import assess_readiness
from etl.assessment.advisor import generate_advice


class PipelineError(Exception):
    pass


def run_pipeline(
    input_csv_path: str,
    output_csv_path: str,
    max_iterations: int = 3
) -> Dict[str, Any]:

    logger = logging.getLogger(__name__)
    logger.debug(
        "Entering run_pipeline: input=%s output=%s max_iter=%s",
        input_csv_path, output_csv_path, max_iterations
    )

    # ---------------------------
    # 1. Read raw input
    # ---------------------------
    df_raw, read_meta = read_structured_safe(input_csv_path)
    df_current = df_raw.copy()
    history = []

    # Initial profile
    profile = ensure_json_serializable(profile_dataframe(df_current))

    # ---------------------------
    # 2. Agent loop (ETL)
    # ---------------------------
    for iteration in range(1, max_iterations + 1):

        feedback = sanitize_feedback(history[-1]) if history else None
        plan = generate_plan(profile, feedback)

        if not plan.get("steps"):
            raise PipelineError("Planner returned empty or invalid steps")
        if not plan["steps"]:
            return {
                "status": "success",
                "iterations": iteration,
                "plan": plan,
                "history": history,
                "read_metadata": read_meta,
            }

        try:
            result = execute_plan(df_current, plan, profile)
            df_next = result["df"]

            validate_transformation(df_current, df_next, plan)

            # Persist cleaned output
            df_next.to_csv(output_csv_path, index=False)

            history.append({
                "iteration": iteration,
                "status": "success",
                "plan": plan,
                "execution_log": result["log"],
            })

            # ---------------------------
            # 3. Post-ETL assessment
            # ---------------------------
            final_profile = ensure_json_serializable(
                profile_dataframe(df_next)
            )

            confidence = compute_confidence(df_next, final_profile)
            readiness = assess_readiness(confidence, final_profile)

            advisor = generate_advice(
                profile=final_profile,
                confidence=confidence,
                readiness=readiness,
            )

            # ---------------------------
            # 4. Final response
            # ---------------------------
            return {
                "status": "success",
                "iterations": iteration,
                "plan": plan,
                "history": history,
                "read_metadata": read_meta,
                "assessment": {
                    "confidence": confidence,
                    "readiness": readiness,
                    "advisor": advisor,
                },
            }

        except Exception as e:
            history.append({
                "iteration": iteration,
                "status": "failed",
                "error": str(e),
                "plan": plan,
            })

            profile["last_failure"] = {
                "error": str(e),
                "failed_plan": plan,
            }

    raise PipelineError("Agent failed to converge after max iterations")
