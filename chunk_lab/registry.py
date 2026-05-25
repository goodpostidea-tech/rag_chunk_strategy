"""Strategy registry."""

from chunk_lab.base import BaseChunkStrategy
from chunk_lab.strategies import (
    ContextualChunkStrategy,
    FixedSizeStrategy,
    MetadataEnrichedStrategy,
    MultiGranularityStrategy,
    ParentChildStrategy,
    PropositionChunkStrategy,
    RecursiveBaselineStrategy,
    SemanticChunkStrategy,
)

STRATEGY_REGISTRY: dict[str, type[BaseChunkStrategy]] = {
    RecursiveBaselineStrategy.name: RecursiveBaselineStrategy,
    FixedSizeStrategy.name: FixedSizeStrategy,
    SemanticChunkStrategy.name: SemanticChunkStrategy,
    ParentChildStrategy.name: ParentChildStrategy,
    MetadataEnrichedStrategy.name: MetadataEnrichedStrategy,
    ContextualChunkStrategy.name: ContextualChunkStrategy,
    PropositionChunkStrategy.name: PropositionChunkStrategy,
    MultiGranularityStrategy.name: MultiGranularityStrategy,
}


def list_strategies() -> list[dict]:
    return [
        {
            "name": cls.name,
            "category": cls.category,
            "description": cls.description,
        }
        for cls in STRATEGY_REGISTRY.values()
    ]


def get_strategy(name: str, **init_kwargs) -> BaseChunkStrategy:
    key = name.strip().lower()
    if key not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY))
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_REGISTRY[key](**init_kwargs)
