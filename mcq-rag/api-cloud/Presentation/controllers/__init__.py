"""
Presentation layer controllers package.
Provides FastAPI controllers for the RAG MCQ system.
"""

from .mcq_controller import MCQController, get_mcq_controller, set_mcq_controller
from .document_controller import DocumentController, get_document_controller, set_document_controller
from .system_controller import SystemController, get_system_controller, set_system_controller

__all__ = [
    "MCQController",
    "get_mcq_controller",
    "set_mcq_controller",
    "DocumentController",
    "get_document_controller",
    "set_document_controller",
    "SystemController",
    "get_system_controller",
    "set_system_controller"
]
