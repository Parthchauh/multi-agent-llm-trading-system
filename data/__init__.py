"""Public market-data API for deterministic strategy research."""

from data.cache import (
    InMemoryMarketDataCache,
    MarketDataCache,
    MarketDataCacheKey,
)
from data.provider import (
    CachedMarketDataProvider,
    MarketDataProvider,
    MarketDataProviderError,
    YahooFinanceProvider,
)
from data.research_split import (
    DatasetFingerprint,
    FinalHoldoutEvaluationView,
    FrozenStrategySelection,
    ResearchDatasetSplit,
    ResearchDevelopmentView,
    ResearchGenerationView,
    ResearchPartition,
    ResearchPartitionRole,
    ResearchSplitError,
)
from data.schema import (
    CANONICAL_OHLCV_COLUMNS,
    DuplicateTimestampError,
    EmptyMarketDataError,
    InvalidMarketDataValueError,
    MarketDataContract,
    MarketDataIndexError,
    MarketDataValidationError,
    MissingMarketDataColumnsError,
    validate_ohlcv,
)

__all__ = [
    "CANONICAL_OHLCV_COLUMNS",
    "CachedMarketDataProvider",
    "DatasetFingerprint",
    "DuplicateTimestampError",
    "EmptyMarketDataError",
    "FinalHoldoutEvaluationView",
    "FrozenStrategySelection",
    "InMemoryMarketDataCache",
    "InvalidMarketDataValueError",
    "MarketDataCache",
    "MarketDataCacheKey",
    "MarketDataContract",
    "MarketDataIndexError",
    "MarketDataProvider",
    "MarketDataProviderError",
    "MarketDataValidationError",
    "MissingMarketDataColumnsError",
    "ResearchDatasetSplit",
    "ResearchDevelopmentView",
    "ResearchGenerationView",
    "ResearchPartition",
    "ResearchPartitionRole",
    "ResearchSplitError",
    "YahooFinanceProvider",
    "validate_ohlcv",
]
