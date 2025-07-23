"""
Enhanced RAG System for Multiple Choice Question Generation
Author: MathematicGuy
Date: July 2025

This module implements an advanced RAG system specifically designed for generating
high-quality Multiple Choice Questions from educational documents.
"""

import os
import json
import time
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import numpy as np

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

# Transformers imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.pipelines import pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig


class QuestionType(Enum):
    """Enumeration of different question types"""
    DEFINITION = "definition"
    COMPARISON = "comparison"
    APPLICATION = "application"
    ANALYSIS = "analysis"
    EVALUATION = "evaluation"


class DifficultyLevel(Enum):
    """Enumeration of difficulty levels"""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class MCQOption:
    """Data class for MCQ options"""
    label: str
    text: str
    is_correct: bool


@dataclass
class MCQQuestion:
    """Data class for Multiple Choice Question"""
    question: str
    context: str
    options: List[MCQOption]
    explanation: str
    difficulty: str
    topic: str
    question_type: str
    source: str
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "question": self.question,
            "context": self.context,
            "options": {opt.label: opt.text for opt in self.options},
            "correct_answer": next(opt.label for opt in self.options if opt.is_correct),
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "topic": self.topic,
            "question_type": self.question_type,
            "source": self.source,
            "confidence_score": self.confidence_score
        }


class PromptTemplateManager:
    """Manages different prompt templates for various question types"""

    def __init__(self):
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different question types"""
        return {
            "base": """
Bạn là một chuyên gia giáo dục và thiết kế câu hỏi. Nhiệm vụ của bạn là tạo ra một câu hỏi trắc nghiệm chất lượng cao từ nội dung được cung cấp.

Yêu cầu:
1. Tạo một câu hỏi rõ ràng, không mơ hồ
2. Cung cấp đúng 4 lựa chọn (A, B, C, D)
3. Chỉ có một đáp án đúng
4. Các phương án sai phải hợp lý nhưng rõ ràng là sai
5. Bao gồm giải thích cho đáp án đúng
6. Xác định mức độ khó và chủ đề

Nội dung: {context}
Chủ đề tập trung: {topic}
Mức độ khó: {difficulty}
Loại câu hỏi: {question_type}

Trả về chỉ dưới dạng JSON hợp lệ:
{{
    "question": "Câu hỏi của bạn",
    "options": {{
        "A": "Lựa chọn A",
        "B": "Lựa chọn B",
        "C": "Lựa chọn C",
        "D": "Lựa chọn D"
    }},
    "correct_answer": "A",
    "explanation": "Giải thích chi tiết",
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "question_type": "{question_type}"
}}
""",

            "definition": """
Tạo câu hỏi trắc nghiệm kiểm tra hiểu biết về định nghĩa thuật ngữ.
Tập trung vào định nghĩa chính xác và những hiểu lầm phổ biến.

{base_template}
""",

            "application": """
Tạo câu hỏi trắc nghiệm dựa trên tình huống thực tế yêu cầu áp dụng khái niệm để giải quyết vấn đề.
Bao gồm các tình huống thực tế nơi khái niệm được sử dụng.

{base_template}
""",

            "analysis": """
Thiết kế câu hỏi yêu cầu phân tích code, sơ đồ hoặc các tình huống phức tạp.
Kiểm tra hiểu biết sâu sắc và tư duy phản biện.

{base_template}
"""
        }

    def get_template(self, question_type: QuestionType = QuestionType.DEFINITION) -> str:
        """Get prompt template for specific question type"""
        return self.templates.get(question_type.value, self.templates["base"])


class QualityValidator:
    """Validates the quality of generated MCQ questions"""

    def __init__(self):
        self.min_question_length = 10
        self.max_question_length = 200
        self.min_explanation_length = 20

    def validate_mcq(self, mcq: MCQQuestion) -> Tuple[bool, List[str]]:
        """Validate MCQ and return validation result with issues"""
        issues = []

        # Check question length
        if len(mcq.question) < self.min_question_length:
            issues.append("Question too short")
        elif len(mcq.question) > self.max_question_length:
            issues.append("Question too long")

        # Check options count
        if len(mcq.options) != 4:
            issues.append("Must have exactly 4 options")

        # Check for single correct answer
        correct_count = sum(1 for opt in mcq.options if opt.is_correct)
        if correct_count != 1:
            issues.append("Must have exactly one correct answer")

        # Check explanation
        if len(mcq.explanation) < self.min_explanation_length:
            issues.append("Explanation too short")

        # Check for distinct options
        option_texts = [opt.text for opt in mcq.options]
        if len(set(option_texts)) != len(option_texts):
            issues.append("Options must be distinct")

        return len(issues) == 0, issues

    def calculate_quality_score(self, mcq: MCQQuestion) -> float:
        """Calculate quality score from 0 to 100"""
        is_valid, issues = self.validate_mcq(mcq)

        if not is_valid:
            return 0.0

        # Start with base score
        score = 70.0

        # Bonus for good explanation length
        if len(mcq.explanation) > 50:
            score += 10

        # Bonus for appropriate question length
        if 20 <= len(mcq.question) <= 100:
            score += 10

        # Bonus for diverse option lengths (indicates good distractors)
        option_lengths = [len(opt.text) for opt in mcq.options]
        if max(option_lengths) - min(option_lengths) < 50:  # Similar lengths
            score += 10

        return min(score, 100.0)


class DifficultyAnalyzer:
    """Analyzes and adjusts question difficulty"""

    def __init__(self):
        self.difficulty_keywords = {
            DifficultyLevel.EASY: ["là gì", "định nghĩa", "ví dụ", "đơn giản"],
            DifficultyLevel.MEDIUM: ["so sánh", "khác biệt", "ứng dụng", "khi nào"],
            DifficultyLevel.HARD: ["phân tích", "đánh giá", "tối ưu", "thiết kế"],
            DifficultyLevel.EXPERT: ["tổng hợp", "sáng tạo", "nghiên cứu", "phát triển"]
        }

    def assess_difficulty(self, question: str, context: str) -> DifficultyLevel:
        """Assess question difficulty based on content analysis"""
        question_lower = question.lower()

        # Count difficulty indicators
        difficulty_scores = {}
        for level, keywords in self.difficulty_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            difficulty_scores[level] = score

        # Return highest scoring difficulty
        if not any(difficulty_scores.values()):
            return DifficultyLevel.MEDIUM

        return max(difficulty_scores.keys(), key=lambda k: difficulty_scores[k])


class ContextAwareRetriever:
    """Enhanced retriever with context awareness and diversity"""

    def __init__(self, vector_db: FAISS, diversity_threshold: float = 0.7):
        self.vector_db = vector_db
        self.diversity_threshold = diversity_threshold

    def retrieve_diverse_contexts(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve documents with semantic diversity"""
        # Get more candidates than needed
        candidates = self.vector_db.similarity_search(query, k=k*2)

        if not candidates:
            return []

        # Select diverse documents
        selected = [candidates[0]]  # Always include the most relevant

        for candidate in candidates[1:]:
            if len(selected) >= k:
                break

            # Check diversity with already selected documents
            is_diverse = True
            for selected_doc in selected:
                similarity = self._calculate_similarity(candidate.page_content,
                                                     selected_doc.page_content)
                if similarity > self.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate)

        return selected[:k]

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (simplified implementation)"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)


class EnhancedRAGMCQGenerator:
    """Enhanced RAG system for MCQ generation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.embeddings = None
        self.llm = None
        self.vector_db = None
        self.retriever = None
        self.prompt_manager = PromptTemplateManager()
        self.validator = QualityValidator()
        self.difficulty_analyzer = DifficultyAnalyzer()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "embedding_model": "bkai-foundation-models/vietnamese-bi-encoder",
            "llm_model": "google/gemma-2b-it",
            "chunk_size": 500,
            "chunk_overlap": 50,
            "retrieval_k": 5,
            "generation_temperature": 0.7,
            "max_tokens": 512,
            "diversity_threshold": 0.7
        }

    def initialize_system(self):
        """Initialize all system components"""
        print("🔧 Initializing Enhanced RAG MCQ Generator...")

        # Load embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"]
        )
        print("✅ Embeddings loaded")

        # Load LLM
        self.llm = self._load_llm()
        print("✅ LLM loaded")

    def _load_llm(self) -> HuggingFacePipeline:
        """Load and configure the LLM"""
        token_path = Path("./api_key/hugging_face_token.txt")
        if token_path.exists():
            with token_path.open("r") as f:
                hf_token = f.read().strip()
        else:
            hf_token = None

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config["llm_model"],
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=hf_token
        )

        tokenizer = AutoTokenizer.from_pretrained(self.config["llm_model"])
        tokenizer.pad_token = tokenizer.eos_token

        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.config["max_tokens"],
            temperature=self.config["generation_temperature"],
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto"
        )

        return HuggingFacePipeline(pipeline=model_pipeline)

    def load_documents(self, folder_path: str) -> Tuple[List[Document], List[str]]:
        """Load and process documents"""
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        pdf_files = list(folder.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {folder}")

        all_docs, filenames = [], []
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                all_docs.extend(docs)
                filenames.append(pdf_file.name)
                print(f"✅ Loaded {pdf_file.name} ({len(docs)} pages)")
            except Exception as e:
                print(f"❌ Failed loading {pdf_file.name}: {e}")

        return all_docs, filenames

    def build_vector_database(self, docs: List[Document]) -> int:
        """Build vector database with semantic chunking"""
        chunker = SemanticChunker(
            embeddings=self.embeddings,
            buffer_size=1,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
            min_chunk_size=self.config["chunk_size"],
            add_start_index=True
        )

        chunks = chunker.split_documents(docs)
        self.vector_db = FAISS.from_documents(chunks, embedding=self.embeddings)

        # Initialize enhanced retriever
        self.retriever = ContextAwareRetriever(
            self.vector_db,
            self.config["diversity_threshold"]
        )

        print(f"✅ Created vector database with {len(chunks)} chunks")
        return len(chunks)

    def generate_mcq(self,
                     topic: str,
                     difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
                     question_type: QuestionType = QuestionType.DEFINITION,
                     context_query: Optional[str] = None) -> MCQQuestion:
        """Generate a single MCQ"""

        if not self.retriever:
            raise RuntimeError("System not initialized. Call initialize_system() first.")

        # Use topic as query if no specific context query provided
        query = context_query or topic

        # Retrieve relevant contexts
        contexts = self.retriever.retrieve_diverse_contexts(
            query, k=self.config["retrieval_k"]
        )

        if not contexts:
            raise ValueError(f"No relevant context found for topic: {topic}")

        # Format contexts
        context_text = "\n\n".join(doc.page_content for doc in contexts)

        # Get appropriate prompt template
        template_text = self.prompt_manager.get_template(question_type)
        prompt_template = PromptTemplate.from_template(template_text)

        # Generate question
        prompt_input = {
            "context": context_text,
            "topic": topic,
            "difficulty": difficulty.value,
            "question_type": question_type.value
        }

        formatted_prompt = prompt_template.format(**prompt_input)
        response = self.llm(formatted_prompt)

        # Parse JSON response
        try:
            # Extract JSON from response
            json_start = response.rfind("{")
            json_end = response.rfind("}") + 1
            json_text = response[json_start:json_end]

            response_data = json.loads(json_text)

            # Create MCQ object
            options = []
            for label, text in response_data["options"].items():
                is_correct = label == response_data["correct_answer"]
                options.append(MCQOption(label, text, is_correct))

            mcq = MCQQuestion(
                question=response_data["question"],
                context=context_text[:500] + "..." if len(context_text) > 500 else context_text,
                options=options,
                explanation=response_data.get("explanation", ""),
                difficulty=difficulty.value,
                topic=topic,
                question_type=question_type.value,
                source=f"{contexts[0].metadata.get('source', 'Unknown')}"
            )

            # Calculate quality score
            mcq.confidence_score = self.validator.calculate_quality_score(mcq)

            return mcq

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def generate_batch(self,
                      topics: List[str],
                      count_per_topic: int = 5,
                      difficulties: Optional[List[DifficultyLevel]] = None,
                      question_types: Optional[List[QuestionType]] = None) -> List[MCQQuestion]:
        """Generate batch of MCQs"""

        if difficulties is None:
            difficulties = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

        if question_types is None:
            question_types = [QuestionType.DEFINITION, QuestionType.APPLICATION]

        mcqs = []
        total_questions = len(topics) * count_per_topic

        print(f"🎯 Generating {total_questions} MCQs...")

        for i, topic in enumerate(topics):
            print(f"📝 Processing topic {i+1}/{len(topics)}: {topic}")

            for j in range(count_per_topic):
                try:
                    # Cycle through difficulties and question types
                    difficulty = difficulties[j % len(difficulties)]
                    question_type = question_types[j % len(question_types)]

                    mcq = self.generate_mcq(topic, difficulty, question_type)
                    mcqs.append(mcq)

                    print(f"  ✅ Generated question {j+1}/{count_per_topic} "
                          f"(Quality: {mcq.confidence_score:.1f})")

                except Exception as e:
                    print(f"  ❌ Failed to generate question {j+1}: {e}")

        print(f"🎉 Generated {len(mcqs)}/{total_questions} MCQs successfully")
        return mcqs

    def export_mcqs(self, mcqs: List[MCQQuestion], output_path: str):
        """Export MCQs to JSON file"""
        output_data = {
            "metadata": {
                "total_questions": len(mcqs),
                "generation_timestamp": time.time(),
                "average_quality": np.mean([mcq.confidence_score for mcq in mcqs])
            },
            "questions": [mcq.to_dict() for mcq in mcqs]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"📁 Exported {len(mcqs)} MCQs to {output_path}")


def main():
    """Main function demonstrating the enhanced RAG MCQ system"""
    print("🚀 Starting Enhanced RAG MCQ Generation System")

    # Initialize system
    generator = EnhancedRAGMCQGenerator()
    generator.initialize_system()

    # Load documents
    folder_path = "pdf_folder"  # Update with your path
    try:
        docs, filenames = generator.load_documents(folder_path)
        num_chunks = generator.build_vector_database(docs)

        print(f"📚 System ready with {len(filenames)} files and {num_chunks} chunks")

        # Generate sample MCQs
        topics = ["Object-Oriented Programming", "Inheritance", "Polymorphism"]

        # Single question generation
        print("\n🎯 Generating single MCQ...")
        mcq = generator.generate_mcq(
            topic="OOP",
            difficulty=DifficultyLevel.MEDIUM,
            question_type=QuestionType.DEFINITION
        )

        print(f"Question: {mcq.question}")
        print(f"Quality Score: {mcq.confidence_score:.1f}")

        # Batch generation
        print("\n🎯 Generating batch MCQs...")
        mcqs = generator.generate_batch(
            topics=topics,
            count_per_topic=2
        )

        # Export results
        output_path = "generated_mcqs.json"
        generator.export_mcqs(mcqs, output_path)

        # Quality summary
        quality_scores = [mcq.confidence_score for mcq in mcqs]
        print(f"\n📊 Quality Summary:")
        print(f"Average Quality: {np.mean(quality_scores):.1f}")
        print(f"Min Quality: {np.min(quality_scores):.1f}")
        print(f"Max Quality: {np.max(quality_scores):.1f}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
