"""
Feedback tracking and self-improvement for built agents.

Reads existing autocomp search run outputs (plans, implementations, evaluation
results, stdout/stderr) to identify patterns and recommend agent config changes.
"""

import json
import re
import yaml
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from autocomp.common import logger, LLMClient


@dataclass
class FeedbackReport:
    """Aggregated feedback from analyzing search run outputs."""
    total_candidates: int = 0
    correct_candidates: int = 0
    improving_candidates: int = 0
    error_patterns: Counter = field(default_factory=Counter)
    optimization_usage: Counter = field(default_factory=Counter)
    optimization_successes: Counter = field(default_factory=Counter)
    novel_optimizations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class FeedbackTracker:
    """
    Analyzes autocomp search run outputs to identify patterns and recommend
    improvements to the agent config (optimization menu, rules).

    Tracks three signals:
    1. Recurring errors -- suggest adding rules to prevent them
    2. Unused optimizations -- flag for review/removal
    3. Discovered optimizations -- suggest adding to menu
    """

    def __init__(self, config_dir: str | Path, llm_client: LLMClient = None):
        self.config_dir = Path(config_dir)
        self.llm_client = llm_client

    def analyze(self, run_dir: str | Path) -> FeedbackReport:
        """
        Analyze a completed search run and produce a feedback report.

        Args:
            run_dir: Path to the search run output directory (contains
                     candidates-iter-N/ subdirectories).
        """
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        report = FeedbackReport()

        # Load the optimization menu for comparison
        menu_options = self._load_menu_options()

        # Walk through iteration directories
        iter_dirs = sorted(run_dir.glob("candidates-iter-*"),
                           key=lambda p: int(p.name.split("-")[-1]))

        for iter_dir in iter_dirs:
            self._analyze_iteration(iter_dir, report, menu_options)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report, menu_options)

        logger.info(
            "FeedbackTracker: analyzed %d candidates (%d correct, %d improving), "
            "%d error patterns, %d optimization types used",
            report.total_candidates, report.correct_candidates,
            report.improving_candidates, len(report.error_patterns),
            len(report.optimization_usage),
        )

        return report

    def apply_recommendations(self, report: FeedbackReport) -> dict[str, str]:
        """
        Use an LLM to apply recommendations to the agent config files.

        Returns a dict of {filename: updated_content} for review.
        Does NOT write files -- caller decides whether to apply.
        """
        if not self.llm_client:
            raise ValueError("LLM client required for apply_recommendations")

        updates: dict[str, str] = {}

        # Load current configs
        menu_path = self.config_dir / "optimization_menu.yaml"
        rules_path = self.config_dir / "rules.yaml"

        current_menu = menu_path.read_text() if menu_path.exists() else ""
        current_rules = rules_path.read_text() if rules_path.exists() else ""

        recommendations_text = "\n".join(f"- {r}" for r in report.recommendations)

        if not recommendations_text.strip():
            logger.info("No recommendations to apply.")
            return updates

        # Update optimization menu
        if any("menu" in r.lower() or "optimization" in r.lower() or "add strategy" in r.lower()
               for r in report.recommendations):
            prompt = f"""Below is the current optimization menu config and a list of recommendations based on search run analysis.

Update the optimization menu YAML to incorporate the relevant recommendations. Preserve the YAML format exactly.

Current optimization_menu.yaml:
```
{current_menu}
```

Recommendations:
{recommendations_text}

Return ONLY the updated YAML content (no markdown fences):"""

            responses = self.llm_client.chat(prompt=prompt, num_candidates=1, temperature=0)
            if responses:
                updated = responses[0].strip()
                updated = updated.strip("`").strip()
                if updated.startswith("yaml"):
                    updated = updated[4:].strip()
                updates["optimization_menu.yaml"] = updated

        # Update rules
        if any("rule" in r.lower() or "constraint" in r.lower() or "error" in r.lower()
               for r in report.recommendations):
            prompt = f"""Below is the current rules config and a list of recommendations based on search run analysis.

Update the rules YAML to incorporate the relevant recommendations. Preserve the YAML format exactly.

Current rules.yaml:
```
{current_rules}
```

Recommendations:
{recommendations_text}

Return ONLY the updated YAML content (no markdown fences):"""

            responses = self.llm_client.chat(prompt=prompt, num_candidates=1, temperature=0)
            if responses:
                updated = responses[0].strip()
                updated = updated.strip("`").strip()
                if updated.startswith("yaml"):
                    updated = updated[4:].strip()
                updates["rules.yaml"] = updated

        return updates

    def write_updates(self, updates: dict[str, str]):
        """Write the updates to disk (after user review)."""
        for filename, content in updates.items():
            path = self.config_dir / filename
            # Back up original
            if path.exists():
                backup = path.with_suffix(path.suffix + ".bak")
                backup.write_text(path.read_text())
                logger.info("Backed up %s to %s", path, backup)
            path.write_text(content)
            logger.info("Updated %s", path)

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _load_menu_options(self) -> list[str]:
        menu_path = self.config_dir / "optimization_menu.yaml"
        if not menu_path.exists():
            return []
        with open(menu_path) as f:
            data = yaml.safe_load(f) or {}
        items = data.get("optimizations", [])
        return [item["strategy"] if isinstance(item, dict) else str(item) for item in items]

    def _analyze_iteration(self, iter_dir: Path, report: FeedbackReport,
                           menu_options: list[str]):
        """Analyze a single iteration directory."""
        # Read candidate files
        for cand_file in iter_dir.glob("candidate_*.txt"):
            report.total_candidates += 1
            try:
                content = cand_file.read_text()
                # Parse the repr'd CodeCandidate for key fields
                score = self._extract_field(content, "score")
                plan = self._extract_field(content, "plan")
                stderr = self._extract_field(content, "stderr")

                if score and score != "inf" and score != "None":
                    report.correct_candidates += 1

                # Track optimization usage from plans
                if plan and plan != "None":
                    matched = self._match_plan_to_menu(plan, menu_options)
                    if matched:
                        for opt in matched:
                            report.optimization_usage[opt] += 1
                            if score and score != "inf" and score != "None":
                                report.optimization_successes[opt] += 1
                    else:
                        # Could be a novel optimization
                        first_line = plan.strip().split("\n")[0][:200]
                        if len(first_line) > 20:
                            report.novel_optimizations.append(first_line)

                # Track error patterns
                if stderr and stderr != "None":
                    patterns = self._extract_error_patterns(stderr)
                    for p in patterns:
                        report.error_patterns[p] += 1

            except Exception as e:
                logger.debug("Failed to parse candidate %s: %s", cand_file, e)

        # Also read plan files directly
        for plan_file in iter_dir.glob("plan_*.txt"):
            try:
                plan_text = plan_file.read_text()
                matched = self._match_plan_to_menu(plan_text, menu_options)
                for opt in matched:
                    report.optimization_usage[opt] += 1
            except Exception:
                pass

    @staticmethod
    def _extract_field(repr_text: str, field_name: str) -> str | None:
        """Extract a field value from a repr'd CodeCandidate string."""
        pattern = rf"{field_name}=['\"]?(.*?)['\"]?[,)]"
        match = re.search(pattern, repr_text, re.DOTALL)
        if match:
            return match.group(1)[:500]
        # Try to find field_name= followed by content
        pattern2 = rf"{field_name}=(.*?)(?:,\s*\w+=|\)$)"
        match2 = re.search(pattern2, repr_text, re.DOTALL)
        if match2:
            return match2.group(1)[:500].strip("'\"")
        return None

    @staticmethod
    def _match_plan_to_menu(plan: str, menu_options: list[str]) -> list[str]:
        """Match a plan to menu options based on keyword overlap."""
        plan_lower = plan.lower()
        matched = []
        for opt in menu_options:
            opt_words = set(re.findall(r"\w{4,}", opt.lower()))
            if not opt_words:
                continue
            matches = sum(1 for w in opt_words if w in plan_lower)
            if matches >= len(opt_words) * 0.4:
                matched.append(opt)
        return matched

    @staticmethod
    def _extract_error_patterns(stderr: str) -> list[str]:
        """Extract error patterns from stderr, normalized for counting."""
        patterns = []
        for line in stderr.split("\\n"):
            line = line.strip()
            if not line:
                continue
            if any(kw in line.lower() for kw in ("error", "exception", "traceback", "failed", "invalid")):
                # Normalize numbers and paths
                normalized = re.sub(r"\d+", "N", line)
                normalized = re.sub(r"/[\w/]+", "PATH", normalized)
                normalized = normalized[:150]
                patterns.append(normalized)
        return patterns[:10]

    def _generate_recommendations(self, report: FeedbackReport,
                                   menu_options: list[str]) -> list[str]:
        """Generate actionable recommendations from the report."""
        recs = []

        # Recurring errors -> suggest rules
        for pattern, count in report.error_patterns.most_common(5):
            if count >= 3:
                recs.append(
                    f"Add a rule to prevent recurring error (seen {count}x): {pattern}"
                )

        # Unused optimizations -> suggest review/removal
        if menu_options and report.optimization_usage:
            never_used = [opt for opt in menu_options
                          if opt not in report.optimization_usage
                          and opt != "Other methods not listed here."]
            if never_used and len(never_used) <= len(menu_options) // 2:
                for opt in never_used[:5]:
                    recs.append(f"Review unused optimization (never selected): {opt}")

        # Optimizations that never improve -> suggest review
        for opt, usage in report.optimization_usage.most_common():
            success = report.optimization_successes.get(opt, 0)
            if usage >= 5 and success == 0:
                recs.append(
                    f"Review ineffective optimization (used {usage}x, never improved): {opt}"
                )

        # Novel optimizations that worked -> suggest adding to menu
        if report.novel_optimizations:
            seen = Counter(report.novel_optimizations)
            for opt, count in seen.most_common(3):
                if count >= 2:
                    recs.append(f"Consider adding new strategy to menu (appeared {count}x): {opt}")

        return recs
