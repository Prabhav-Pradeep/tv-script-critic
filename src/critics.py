"""Critic building blocks shared by the gold critic and the proxy critics.

Kept in its own module so a saved critic (joblib) can be loaded from any script:
pickle stores the class by module path, and a class defined in a script run as
__main__ cannot be found again from another script.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from text_utils import mask_names

LEX_ALPHA = 10     # chosen by season-grouped CV in critic_baseline.py --mask


def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, sublinear_tf=True,
                           max_features=50_000, lowercase=True)


class LexicalCritic:
    """TF-IDF + ridge on (already masked) text."""

    def __init__(self, alpha=LEX_ALPHA):
        self.alpha = alpha

    def fit(self, texts, y):
        self.vec = make_vectorizer().fit(texts)
        self.ridge = Ridge(alpha=self.alpha).fit(self.vec.transform(texts), y)
        return self

    def predict(self, texts):
        return self.ridge.predict(self.vec.transform(texts))


class SemanticCritic:
    """Standardised sentence embeddings + ridge."""

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, emb, y):
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha)).fit(emb, y)
        return self

    def predict(self, emb):
        return self.model.predict(emb)


class ProxyEnsemble:
    """K lexical critics, each trained on its own random subset, with gold scores as labels.

    score(raw_texts) -> (n_texts, K) array, one column per proxy. Selection rules (single,
    mean, mean - lambda * std, min) are applied to these columns by the experiment.
    """

    def __init__(self, names, members, subset_size):
        self.names = names
        self.members = members
        self.subset_size = subset_size

    def score(self, texts):
        masked = [mask_names(t, self.names) for t in texts]
        return np.column_stack([m.predict(masked) for m in self.members])


class GoldCritic:
    """Hidden 'truth' of the experiment: lexical + semantic critics, stacked.

    score(raw_texts) repeats the exact training steps: mask names -> both critics ->
    weighted sum. The sentence-transformer itself is not pickled; it is reloaded by name.
    """

    def __init__(self, names, lexical, semantic, weights, intercept, embedder_name):
        self.names = names
        self.lexical = lexical
        self.semantic = semantic
        self.weights = np.asarray(weights)
        self.intercept = float(intercept)
        self.embedder_name = embedder_name
        self._embedder = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_embedder"] = None
        return state

    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder

    def score(self, texts, emb=None):
        """Gold score for raw (unmasked) texts. Pass `emb` to reuse precomputed embeddings."""
        from embed_chunks import embed_chunks
        masked = [mask_names(t, self.names) for t in texts]
        if emb is None:
            emb, _ = embed_chunks(masked, self.embedder())
        parts = np.column_stack([self.lexical.predict(masked), self.semantic.predict(emb)])
        return parts @ self.weights + self.intercept
