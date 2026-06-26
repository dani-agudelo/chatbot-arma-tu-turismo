"""Parsers de nodos: bloques por estructura del PDF y subdivision por tamaño."""

from __future__ import annotations

import re
from typing import Sequence

from llama_index.core.node_parser import NodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, TextNode
from config import CHUNK_OVERLAP, CHUNK_SIZE

# Encabezados frecuentes en documentos normativos (solo patron, sin nombres en variables de negocio).
_STRUCTURE_HEADING_PATTERN = re.compile(
    r"(?m)^(?=(?:Artículo|ARTÍCULO|Capítulo|CAPÍTULO|Sección|SECCIÓN)\s+)"
)


def split_text_into_blocks(text: str) -> list[str]:
    """Divide texto en bloques por encabezados o, en su defecto, por párrafos."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    parts = _STRUCTURE_HEADING_PATTERN.split(normalized)
    blocks = [part.strip() for part in parts if part and part.strip()]

    if not blocks:
        blocks = [
            part.strip()
            for part in re.split(r"\n\s*\n+", normalized)
            if part.strip()
        ]

    return blocks


class StructuredParagraphNodeParser(NodeParser):
    """Respeta bloques estructurados y aplica SentenceSplitter solo si hace falta."""

    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: object,
    ) -> list[BaseNode]:
        sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        output_nodes: list[BaseNode] = []

        for node in nodes:
            text = node.get_content(metadata_mode="none")
            metadata = dict(node.metadata or {})
            blocks = split_text_into_blocks(text)

            for block in blocks:
                block_nodes = sentence_splitter.get_nodes_from_documents(
                    [Document(text=block, metadata=metadata)]
                )
                for block_node in block_nodes:
                    if isinstance(block_node, TextNode):
                        block_node.metadata.update(metadata)
                    output_nodes.append(block_node)

        return output_nodes


def get_document_node_parser() -> StructuredParagraphNodeParser:
    """Parser usado en la pipeline de ingesta."""
    return StructuredParagraphNodeParser(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
