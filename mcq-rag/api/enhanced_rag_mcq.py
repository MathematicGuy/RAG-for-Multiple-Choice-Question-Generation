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
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document
from unsloth import FastLanguageModel

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
        self.base_template = self._create_base_template()
        self.templates = self._initialize_templates()

    def _create_base_template(self) -> str:
        """Create the base template used by all question types"""
        return """
            Hãy tạo 1 câu hỏi trắc nghiệm dựa trên nội dung sau đây.

            Nội dung: {context}
            Chủ đề: {topic}
            Mức độ: {difficulty}
            Loại câu hỏi: {question_type}

            QUAN TRỌNG: Chỉ trả về JSON hợp lệ, không có text bổ sung, luôn trả lời bằng tiếng việt:

            {{
                "question": "Câu hỏi rõ ràng về {topic}",
                "options": {{
                    "A": "Đáp án A",
                    "B": "Đáp án B",
                    "C": "Đáp án C",
                    "D": "Đáp án D"
                }},
                "correct_answer": "A",
                "explanation": "Giải thích tại sao đáp án A đúng",
                "topic": "{topic}",
                "difficulty": "{difficulty}",
                "question_type": "{question_type}"
            }}

            Trả lời:
        """

    def _create_question_specific_instruction(self, question_type: str) -> str:
        """Create specific instructions for each question type"""
        instructions = {
            "definition": "Tạo câu hỏi định nghĩa thuật ngữ. Tập trung vào định nghĩa chính xác.",
            "application": "Tạo câu hỏi ứng dụng thực tế. Bao gồm tình huống cụ thể.",
            "analysis": "Tạo câu hỏi phân tích code/sơ đồ. Kiểm tra tư duy phản biện.",
            "comparison": "Tạo câu hỏi so sánh khái niệm. Tập trung vào điểm khác biệt.",
            "evaluation": "Tạo câu hỏi đánh giá phương pháp. Yêu cầu quyết định dựa trên tiêu chí."
        }
        return instructions.get(question_type, "Tạo câu hỏi chất lượng cao.")

    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize all templates using the base template"""
        templates = {"base": self.base_template}

        # Create shorter, more direct templates for each question type
        instructions = {
            "definition": "Tạo câu hỏi định nghĩa thuật ngữ.",
            "application": "Tạo câu hỏi ứng dụng thực tế.",
            "analysis": "Tạo câu hỏi phân tích code/sơ đồ.",
            "comparison": "Tạo câu hỏi so sánh khái niệm.",
            "evaluation": "Tạo câu hỏi đánh giá phương pháp."
        }

        # Add specific templates for each question type
        for question_type in QuestionType:
            instruction = instructions.get(question_type.value, "Tạo câu hỏi chất lượng cao.")
            templates[question_type.value] = f"{instruction}\n\n{self.base_template}"

        return templates

    def get_template(self, question_type: QuestionType = QuestionType.DEFINITION) -> str:
        """Get prompt template for specific question type"""
        return self.templates.get(question_type.value, self.templates["base"])

    def update_base_template(self, new_base_template: str):
        """Update the base template and regenerate all templates"""
        self.base_template = new_base_template
        self.templates = self._initialize_templates()
        print("✅ Base template updated and all templates regenerated")

    def get_template_info(self) -> Dict[str, int]:
        """Get information about all templates (for debugging)"""
        return {
            template_type: len(template)
            for template_type, template in self.templates.items()
        }

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
        print("MCQ Output Quality Score:", issues)

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
            # "llm_model": "unsloth/Qwen2.5-3B-Instruct", # 7B, 1.5B

            "chunk_size": 500,
            "chunk_overlap": 50,
            "retrieval_k": 3,
            "generation_temperature": 0.7,
            "max_tokens": 512,
            "diversity_threshold": 0.7,
            "max_context_length": 600,  # Maximum context characters
            "max_input_tokens": 1600  # Maximum total input tokens
        }

    def _truncate_context(self, context: str, max_length: Optional[int] = None) -> str:
        """Intelligently truncate context to fit within token limits"""
        actual_max_length = max_length if max_length is not None else self.config["max_context_length"]

        if len(context) <= actual_max_length:
            return context

        # Try to truncate at sentence boundary
        sentences = context.split('. ')
        truncated = ""

        for sentence in sentences:
            if len(truncated + sentence + '. ') <= actual_max_length:
                truncated += sentence + '. '
            else:
                break

        # If no complete sentences fit, truncate at word boundary
        if not truncated:
            words = context.split()
            truncated = ""
            for word in words:
                if len(truncated + word + ' ') <= actual_max_length:
                    truncated += word + ' '
                else:
                    break

        return truncated.strip()

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for Vietnamese text (approximation)"""
        # Vietnamese typically has ~0.75 tokens per character
        return int(len(text) * 0.75)

	#? Parse Json String
    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from LLM response with multiple fallback strategies"""
        import re

        # Strategy 1: Clean response of prompt repetition
        clean_response = response
        if "Tạo câu hỏi" in response:
            # Find where the actual response starts (after the prompt)
            response_parts = response.split("JSON:")
            if len(response_parts) > 1:
                clean_response = response_parts[-1].strip()
            else:
                # Try splitting on common phrases
                for split_phrase in ["QUAN TRỌNG:", "Trả về JSON:", "{"]:
                    if split_phrase in response:
                        clean_response = response.split(split_phrase)[-1].strip()
                        if split_phrase == "{":
                            clean_response = "{" + clean_response
                        break

        # Strategy 2: Find JSON boundaries
        json_start = clean_response.find("{")
        json_end = clean_response.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_text = clean_response[json_start:json_end]
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Use regex to find JSON-like structures
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        json_matches = re.findall(json_pattern, clean_response, re.DOTALL)

        for json_match in reversed(json_matches):  # Try from last to first
            try:
                return json.loads(json_match)
            except json.JSONDecodeError:
                continue

        # Strategy 4: Try to fix common JSON issues
        for json_match in reversed(json_matches):
            try:
                # Fix common issues like trailing commas
                fixed_json = re.sub(r',(\s*[}\]])', r'\1', json_match)
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                continue

        raise ValueError("No valid JSON found in response")

    def check_prompt_length(self, prompt: str) -> Tuple[bool, int]:
        """Check if prompt length is within safe limits"""
        estimated_tokens = self._estimate_token_count(prompt)
        max_safe_tokens = self.config["max_input_tokens"]

        is_safe = estimated_tokens <= max_safe_tokens
        return is_safe, estimated_tokens

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
        token_path = Path("./tokens/hugging_face_token.txt")
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

        model, tokenizer = FastLanguageModel.from_pretrained(
            self.config["llm_model"],
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            token=hf_token
        )

        # tokenizer = AutoTokenizer.from_pretrained(self.config["llm_model"])
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
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized. Call initialize_system() first.")

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
        """Generate a single MCQ with proper length management"""

        if not self.retriever:
            raise RuntimeError("System not initialized. Call initialize_system() first.")

        if not self.llm:
            raise RuntimeError("LLM not initialized. Call initialize_system() first.")

        # Use topic as query if no specific context query provided
        query = context_query or topic

        # Retrieve relevant contexts (reduced number)
        contexts = self.retriever.retrieve_diverse_contexts(
            query, k=self.config["retrieval_k"]
        )

        if not contexts:
            raise ValueError(f"No relevant context found for topic: {topic}")

        # Format and truncate contexts
        context_text = "\n\n".join(doc.page_content for doc in contexts)
        context_text = self._truncate_context(context_text)

        # Get appropriate prompt template
        template_text = self.prompt_manager.get_template(question_type)
        template_infor_debug = self.prompt_manager.get_template_info()
        prompt_template = PromptTemplate.from_template(template_text)
        print(f"Template Structure Info: \n {template_infor_debug}")

        # Generate question with length checking
        prompt_input = {
            "context": context_text,
            "topic": topic,
            "difficulty": difficulty.value,
            "question_type": question_type.value
        }

        formatted_prompt = prompt_template.format(**prompt_input)

        # Check prompt length and truncate if necessary
        is_safe, token_count = self.check_prompt_length(formatted_prompt)

        if not is_safe:
            print(f"⚠️ Prompt too long ({token_count} tokens), truncating context...")
            # Further reduce context
            reduced_context = self._truncate_context(context_text, 400)
            prompt_input["context"] = reduced_context
            formatted_prompt = prompt_template.format(**prompt_input)
            is_safe, token_count = self.check_prompt_length(formatted_prompt)

            if not is_safe:
                print(f"⚠️ Still too long ({token_count} tokens), using minimal context...")
                # Use only first paragraph
                minimal_context = context_text.split('\n')[0][:300]
                prompt_input["context"] = minimal_context
                formatted_prompt = prompt_template.format(**prompt_input)

        print(f"📏 Prompt length: {self._estimate_token_count(formatted_prompt)} tokens")

        try:
            response = self.llm.invoke(formatted_prompt)
            print(f"✅ Generated response length: {len(response)} characters")
        except Exception as e:
            if "length" in str(e).lower():
                print(f"❌ Length error persists: {e}")
                # Emergency fallback - use very short context
                emergency_context = topic + ": " + context_text[:200]
                prompt_input["context"] = emergency_context
                formatted_prompt = prompt_template.format(**prompt_input)
                print(f"🚨 Emergency context length: {self._estimate_token_count(formatted_prompt)} tokens")
                response = self.llm.invoke(formatted_prompt)
            else:
                raise e

        # Parse JSON response
        try:
            print(f"🔍 Parsing response (first 300 chars): {response}...")
            response_data = self._extract_json_from_response(response)
            print(f"✅ Successfully parsed JSON response")

            # Create MCQ object
            options = []
            for label, text in response_data["options"].items():
                is_correct = label == response_data["correct_answer"]
                options.append(MCQOption(label, text, is_correct))

            mcq = MCQQuestion(
                question=response_data["question"],
                context=prompt_input["context"],  # Use the truncated context
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
            print(f"❌ Response parsing error: {e}")
            print(f"Raw response: {response[:500]}...")
            raise ValueError(f"Failed to parse LLM response: {e}")

    def generate_batch(self,
                      topics: List[str],
                      question_per_topic: int = 5,
                      difficulties: Optional[List[DifficultyLevel]] = None,
                      question_types: Optional[List[QuestionType]] = None) -> List[MCQQuestion]:
        """Generate batch of MCQs"""

        if difficulties is None:
            difficulties = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

        if question_types is None:
            question_types = [QuestionType.DEFINITION, QuestionType.APPLICATION]

        mcqs = []
        total_questions = len(topics) * question_per_topic

        print(f"🎯 Generating {total_questions} MCQs...")

        for i, topic in enumerate(topics):
            print(f"📝 Processing topic {i+1}/{len(topics)}: {topic}")

            for j in range(question_per_topic):
                try:
                    # Cycle through difficulties and question types
                    difficulty = difficulties[j % len(difficulties)]
                    question_type = question_types[j % len(question_types)]

                    mcq = self.generate_mcq(topic, difficulty, question_type)
                    mcqs.append(mcq)

                    print(f"  ✅ Generated question {j+1}/{question_per_topic} "
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
                "average_quality": np.mean([mcq.confidence_score for mcq in mcqs]) if mcqs else 0
            },
            "questions": [mcq.to_dict() for mcq in mcqs]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"📁 Exported {len(mcqs)} MCQs to {output_path}")

    def debug_system_state(self):
        """Debug function to check system initialization state"""
        print("🔍 System Debug Information:")
        print(f"  Embeddings initialized: {'✅' if self.embeddings else '❌'}")
        print(f"  LLM initialized: {'✅' if self.llm else '❌'}")
        print(f"  Vector database created: {'✅' if self.vector_db else '❌'}")
        print(f"  Retriever initialized: {'✅' if self.retriever else '❌'}")
        print(f"  Config loaded: {'✅' if self.config else '❌'}")

        if self.config:
            print(f"  Embedding model: {self.config.get('embedding_model', 'Not set')}")
            print(f"  LLM model: {self.config.get('llm_model', 'Not set')}")
            print(f"  Max context length: {self.config.get('max_context_length', 'Not set')}")
            print(f"  Max input tokens: {self.config.get('max_input_tokens', 'Not set')}")

        # Show template information
        template_info = self.prompt_manager.get_template_info()
        print(f"  Template sizes:")
        for template_type, size in template_info.items():
            print(f"    {template_type}: {size} characters")


def debug_prompt_templates():
    """Debug function to test prompt template generation"""
    print("🔍 Testing Prompt Templates:")
    prompt_manager = PromptTemplateManager()

    for question_type in QuestionType:
        try:
            template = prompt_manager.get_template(question_type)
            print(f"  {question_type.value}: ✅ Template loaded ({len(template)} chars)")
        except Exception as e:
            print(f"  {question_type.value}: ❌ Error - {e}")


def main():
    """Main function demonstrating the enhanced RAG MCQ system"""
    print("🚀 Starting Enhanced RAG MCQ Generation System")


    # Test prompt templates first
    print("\n🧪 Testing prompt templates...")
    debug_prompt_templates()

    # Initialize system
    generator = EnhancedRAGMCQGenerator()

    # Check initial state
    print("\n🔍 Initial system state:")
    generator.debug_system_state()

    try:
        generator.initialize_system()

        # Check state after initialization
        print("\n🔍 Post-initialization state:")
        generator.debug_system_state()

        # Load documents
        folder_path = "pdfs"  # Updated path to your PDF folder
        try:
            docs, filenames = generator.load_documents(folder_path)
            num_chunks = generator.build_vector_database(docs)
            start = time.time() #? Calc generation time

            print(f"⏱️ Loading Time: {time.time() - start:.2f}s") #? Loading document time
            print(f"📚 System ready with {len(filenames)} files and {num_chunks} chunks")

            # Generate sample MCQs
            topics = ["Object Oriented Programming", "Malware Reverse Engineering"]

            # Single question generation
            # print("\n🎯 Generating single MCQ...")
            # mcq = generator.generate_mcq(
            #     topic=topics[0],
            #     difficulty=DifficultyLevel.MEDIUM,
            #     question_type=QuestionType.DEFINITION
            # )

            # print(f"Question: {mcq.question}")
            # print(f"Quality Score: {mcq.confidence_score:.1f}")

            # Batch generation
            n_question = 2
            print("\n🎯 Generating batch MCQs...")

            #? MAIN OUTPUT: Multiple Choice Question
            mcqs = generator.generate_batch(
                topics=topics,
                question_per_topic=n_question
            )

            # Export results
            output_path = "generated_mcqs.json"
            generator.export_mcqs(mcqs, output_path)

            # Quality summary
            print(f"Average mcq generation time taken: {((time.time() - start)/n_question)/60:.2f} min")
            quality_scores = [mcq.confidence_score for mcq in mcqs]
            print(f"\n📊 Quality Summary:")
            print(f"Average Quality: {np.mean(quality_scores):.1f}")
            print(f"Min Quality: {np.min(quality_scores):.1f}")
            print(f"Max Quality: {np.max(quality_scores):.1f}")

        except FileNotFoundError as e:
            print(f"❌ Document folder error: {e}")
            print("💡 Please ensure your PDF files are in the 'pdfs' folder")
        except Exception as e:
            print(f"❌ Document processing error: {e}")
            print("💡 Check your PDF files and folder structure")

    except Exception as e:
        print(f"❌ System initialization error: {e}")
        print("💡 Check your dependencies and API keys")
        generator.debug_system_state()


if __name__ == "__main__":
    # Check system components
    # generator = EnhancedRAGMCQGenerator()
    # generator.debug_system_state()

    # # Test templates separately
    # debug_prompt_templates()

    main()
