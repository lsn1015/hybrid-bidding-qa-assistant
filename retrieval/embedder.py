from __future__ import annotations

"""
Embedding 模型封装。

默认使用 `config/model.yaml` 中的 `Embedding.model_name`
（如 BAAI/bge-large-zh-v1.5），通过 sentence-transformers 加载。

这里只做一个薄封装，真正的模型下载/缓存由 sentence-transformers 负责。
"""

from pathlib import Path
from typing import Iterable, List

import yaml

try:  # 可选依赖，便于在无 GPU 的环境下安装
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - 环境可能未安装
    SentenceTransformer = None  # type: ignore

from ..logging.logger import get_logger

logger = get_logger(__name__)


class Embedder:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        model_cfg_path = project_root / "config" / "model.yaml"

        with model_cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        emb_cfg = cfg.get("Embedding", {})
        self.model_name: str = emb_cfg.get(
            "model_name", "BAAI/bge-large-zh-v1.5"
        )

        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not available, please `pip install sentence-transformers`."
            )

        logger.info("loading_embedding_model", model=self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        """
        将一组文本编码为向量。
        """
        texts_list = list(texts)
        if not texts_list:
            return []
        embeddings = self._model.encode(
            texts_list, convert_to_numpy=False, normalize_embeddings=True
        )
        return [e.tolist() for e in embeddings]


__all__ = ["Embedder"]

