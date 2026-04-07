"""
Agent assembler for the Agent Builder.

Writes synthesized components to human-editable config files in a directory
that can be loaded by BuiltLLMAgent at runtime.
"""

import datetime
import yaml
from pathlib import Path

from autocomp.common import logger
from autocomp.agent_builder.synthesizer import SynthesizedComponents


class AgentAssembler:
    """Writes synthesized components to an agent config directory."""

    def assemble(
        self,
        components: SynthesizedComponents,
        agent_name: str,
        output_dir: str | Path,
        build_metadata: dict | None = None,
    ) -> Path:
        """
        Write all components to the output directory.

        Returns the output directory path.
        """
        out = Path(output_dir)
        if out.exists():
            logger.info("Output directory %s already exists -- files will be overwritten", out)
        out.mkdir(parents=True, exist_ok=True)

        # agent_config.yaml
        config: dict = {
            "agent_name": agent_name,
            "version": "1.0",
            "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        if build_metadata:
            config["build"] = build_metadata
        config["description"] = f"Auto-generated agent config for {agent_name}"
        self._write_yaml(out / "agent_config.yaml", config)

        # architecture.md
        (out / "architecture.md").write_text(components.architecture_summary)
        logger.info("Wrote architecture.md (%d chars)", len(components.architecture_summary))

        # isa_docs.md
        (out / "isa_docs.md").write_text(components.isa_docs)
        logger.info("Wrote isa_docs.md (%d chars)", len(components.isa_docs))

        # code_examples.md
        if components.code_examples:
            (out / "code_examples.md").write_text(components.code_examples)
            logger.info("Wrote code_examples.md (%d chars)", len(components.code_examples))
        else:
            logger.info("Skipping code_examples.md -- no examples were extracted")

        # optimization_menu.yaml
        menu_items = []
        for opt in components.optimization_menu:
            menu_items.append({"strategy": opt})
        self._write_yaml(out / "optimization_menu.yaml", {"optimizations": menu_items})
        logger.info("Wrote optimization_menu.yaml (%d strategies)", len(menu_items))

        # translate_menu.yaml
        if components.translate_menu:
            translate_items = [{"strategy": s} for s in components.translate_menu]
            self._write_yaml(out / "translate_menu.yaml", {"strategies": translate_items})
            logger.info("Wrote translate_menu.yaml (%d strategies)", len(translate_items))
        else:
            logger.info("Skipping translate_menu.yaml -- no translate strategies generated")

        # rules.yaml
        self._write_yaml(out / "rules.yaml", components.rules)
        n_rules = sum(len(v) for v in components.rules.values() if isinstance(v, list))
        logger.info("Wrote rules.yaml (%d rules across %d categories)", n_rules, len(components.rules))

        logger.info("Agent config assembled at %s", out)
        return out

    @staticmethod
    def _write_yaml(path: Path, data: dict):
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)
