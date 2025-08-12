"""
Presentation layer package.
Provides controllers and API endpoints for the RAG MCQ system.
"""

from .controllers import (
    MCQController,
    DocumentController,
    SystemController,
    get_mcq_controller,
    get_document_controller,
    get_system_controller,
    set_mcq_controller,
    set_document_controller,
    set_system_controller
)

__all__ = [
    "MCQController",
    "DocumentController",
    "SystemController",
    "get_mcq_controller",
    "get_document_controller",
    "get_system_controller",
    "set_mcq_controller",
    "set_document_controller",
    "set_system_controller"
]
