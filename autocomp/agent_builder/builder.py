"""
Top-level AgentBuilder that orchestrates ingest -> synthesize -> assemble.
"""

from pathlib import Path

from autocomp.common import logger, LLMClient
from autocomp.agent_builder.ingestor import KnowledgeIngestor
from autocomp.agent_builder.synthesizer import ComponentSynthesizer
from autocomp.agent_builder.assembler import AgentAssembler


class AgentBuilder:
    """
    Builds a new hardware-target agent from diverse knowledge sources.

    Usage:
        builder = AgentBuilder(llm_model="anthropic::claude-sonnet-4-20250514")
        builder.add_source("directory", path="/path/to/hw-sdk/docs")
        builder.add_source("pdf", path="/path/to/architecture-manual.pdf")
        builder.add_source("webpage", url="https://docs.example.com/isa-reference")
        builder.add_source("github", url="https://github.com/org/hw-examples")

        config_dir = builder.build(
            agent_name="my_accelerator",
            output_dir="autocomp/agents/my_accelerator"
        )
    """

    def __init__(self, llm_model: str, light_llm_model: str | None = None,
                 description: str = ""):
        """
        Args:
            llm_model: Model identifier for synthesis LLM calls.
                       Supports "provider::model" syntax (e.g., "anthropic::claude-sonnet-4-20250514")
                       or just the model name for auto-detection.
            light_llm_model: Optional cheaper/faster model for high-token extraction tasks.
                             Uses the same "provider::model" syntax. Falls back to llm_model if not set.
            description: User-provided context about what the agent is for.
                         This is prepended to every LLM prompt and strongly
                         influences content routing, ISA filtering, and strategy
                         generation. Be specific about:
                         - What level of code the agent optimizes (e.g., kernels,
                           operators, full models)
                         - The programming interface (e.g., NKI, CUDA, HLO)
                         - What's out of scope (e.g., deployment, serving, distributed)
                         Example: "Optimizing NKI kernel code on AWS Trainium 1.
                         The agent rewrites single-kernel source code for better
                         performance. Model-level concerns like sharding, serving,
                         and distributed training are out of scope."
        """
        if "::" in llm_model:
            provider, model = llm_model.split("::", 1)
        else:
            provider = None
            model = llm_model
        self._llm_client = LLMClient(model, provider)

        self._light_llm_client: LLMClient | None = None
        if light_llm_model:
            if "::" in light_llm_model:
                lp, lm = light_llm_model.split("::", 1)
            else:
                lp, lm = None, light_llm_model
            self._light_llm_client = LLMClient(lm, lp)

        self._description = description
        self._ingestor = KnowledgeIngestor()

    def add_source(self, source_type: str, **kwargs):
        """
        Add a knowledge source to ingest.

        Args:
            source_type: One of "directory", "github", "pdf", "webpage", "confluence"
            **kwargs: Source-specific arguments (path, url, email, api_token, etc.)
        """
        self._ingestor.add_source(source_type, **kwargs)

    def build(self, agent_name: str, output_dir: str | Path) -> Path:
        """
        Run the full build pipeline: ingest -> synthesize -> assemble.

        Args:
            agent_name: Name for the new agent.
            output_dir: Directory to write config files to.

        Returns:
            Path to the output directory containing the agent config.
        """
        logger.info("AgentBuilder: starting build for '%s'", agent_name)

        # Stage 1: Ingest
        logger.info("AgentBuilder: Stage 1 -- ingesting knowledge sources")
        indices = self._ingestor.ingest()
        if not indices:
            raise ValueError("No knowledge sources were added. Call add_source() first.")
        logger.info("AgentBuilder: ingested %d sources", len(indices))

        # Stage 2: Synthesize
        logger.info("AgentBuilder: Stage 2 -- synthesizing components")
        synthesizer = ComponentSynthesizer(
            self._llm_client, self._light_llm_client, description=self._description,
        )
        components = synthesizer.synthesize(indices)

        # Stage 3: Assemble
        logger.info("AgentBuilder: Stage 3 -- assembling agent config")
        assembler = AgentAssembler()
        config_dir = assembler.assemble(components, agent_name, output_dir)

        logger.info(
            "AgentBuilder: build complete. Config at %s\n"
            "  - architecture.md (%d chars)\n"
            "  - isa_docs.md (%d chars)\n"
            "  - code_examples.md (%d chars)\n"
            "  - optimization_menu.yaml (%d strategies)\n"
            "  - rules.yaml\n"
            "Review and edit these files before using the agent.",
            config_dir,
            len(components.architecture_summary),
            len(components.isa_docs),
            len(components.code_examples),
            len(components.optimization_menu),
        )

        return config_dir
