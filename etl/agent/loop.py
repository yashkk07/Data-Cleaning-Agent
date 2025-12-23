from typing import Dict, Any, List
import pandas as pd

from etl.profile.profiler import profile_dataframe
from etl.profile.serializer import ensure_json_serializable
from etl.llm.planner import generate_plan
from etl.executor.tool_executor import execute_plan
from etl.validate.validator import validate_transformation


class AgentFailure(Exception):
    pass


def run_agent_loop(
    df_raw: pd.DataFrame,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    Iterative agent loop with reflection.
    Returns final dataframe + execution history.
    """

    history: List[Dict[str, Any]] = []
    df_current = df_raw.copy()

    profile = ensure_json_serializable(profile_dataframe(df_current))

    for iteration in range(1, max_iterations + 1):

        plan = generate_plan(
            profile=profile,
            feedback=history[-1] if history else None
        )

        try:
            df_next = execute_plan(df_current, plan)
            validate_transformation(df_current, df_next)

            history.append({
                "iteration": iteration,
                "plan": plan,
                "status": "success"
            })

            return {
                "df": df_next,
                "history": history
            }

        except Exception as e:
            history.append({
                "iteration": iteration,
                "plan": plan,
                "status": "failed",
                "error": str(e)
            })

            # Reflect: augment profile with failure context
            profile["last_failure"] = {
                "iteration": iteration,
                "error": str(e),
                "failed_plan": plan
            }

    raise AgentFailure("Agent failed to converge after max iterations")
