import sys
sys.path.append("nlp\\language_modeling")
from language_modeling import (
    LanguageModelingDatasetC4,
    LanguageModelingDatasetPTB,
    LanguageModelingDatasetWikitext2,
    LanguageModelingDatasetWikiText103,
)

dataset_factory = {
    # LM dataset
    "wikitext2": LanguageModelingDatasetWikitext2,
    "wikitext103": LanguageModelingDatasetWikiText103,
    "c4": LanguageModelingDatasetC4,
    "ptb": LanguageModelingDatasetPTB,
}
