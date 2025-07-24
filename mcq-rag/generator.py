from verify import QualityValidator, DifficultyAnalyzer
from prompt import PromptTemplateManager
from util import ContextAwareRetriever, DifficultyLevel, QuestionType, MCQQuestion, MCQOption
from typing import Dict, Any, Optional, Tuple, List
import json
import torch
from pathlib import Path
import time
import numpy as np
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
from unsloth import FastLanguageModel
from transformers.pipelines import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import PromptTemplate

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
            "llm_model": "unsloth/Qwen2.5-3B", # 7B, 1.5B
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
