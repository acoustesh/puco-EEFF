"""Type definitions for similarity detection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FunctionInfo:
    """Information about an extracted function.

    Attributes
    ----------
        name: Function or class name
        file: Source file name (relative path)
        start_line: Starting line number in source
        end_line: Ending line number in source
        loc: Lines of code
        hash: AST-based hash (ignores comments, used for AST embeddings)
        text: Full raw source text
        text_for_embedding: Signature + docstring + comment lines (for text embeddings)
        text_hash: Hash of text_for_embedding (used for text embeddings)
        ast_text: AST unparsed code via ast.unparse() (for AST embeddings)
        embedding: Current embedding vector (set during similarity checks)
    """

    name: str
    file: str
    start_line: int
    end_line: int
    loc: int
    hash: str  # AST-based hash (ignores comments)
    text: str  # Full raw source
    text_for_embedding: str = ""  # Signature + docstring + comments
    text_hash: str = ""  # Hash of text_for_embedding
    ast_text: str = ""  # ast.unparse() output
    embedding: list[float] | None = field(default=None, repr=False)
