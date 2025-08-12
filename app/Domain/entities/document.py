"""
Document domain entity.
Contains the core business entity for document handling.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class Document:
    """Domain entity representing a document"""
    content: str
    metadata: Dict[str, Any]
    source: str

    def __post_init__(self):
        """Validate the document after initialization"""
        if not self.content or not self.content.strip():
            raise ValueError("Document content cannot be empty")

        if not self.source:
            raise ValueError("Document source cannot be empty")

        # Ensure metadata is not None
        if self.metadata is None:
            self.metadata = {}

    @property
    def filename(self) -> Optional[str]:
        """Get the filename from source if it's a file path"""
        try:
            return Path(self.source).name
        except Exception:
            return None

    @property
    def page_number(self) -> Optional[int]:
        """Get page number from metadata if available"""
        return self.metadata.get('page', None)

    @property
    def document_type(self) -> str:
        """Get document type from metadata or infer from source"""
        if 'type' in self.metadata:
            return self.metadata['type']

        if self.source.lower().endswith('.pdf'):
            return 'pdf'
        elif self.source.lower().endswith('.docx'):
            return 'docx'
        elif self.source.lower().endswith('.txt'):
            return 'txt'
        else:
            return 'unknown'

    def get_char_count(self) -> int:
        """Get character count of the document content"""
        return len(self.content)

    def get_word_count(self) -> int:
        """Get word count of the document content"""
        return len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary format"""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
            "filename": self.filename,
            "page_number": self.page_number,
            "document_type": self.document_type,
            "char_count": self.get_char_count(),
            "word_count": self.get_word_count()
        }
