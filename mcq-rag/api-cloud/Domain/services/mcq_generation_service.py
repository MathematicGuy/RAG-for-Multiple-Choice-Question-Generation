"""
Core MCQ generation service.
Contains the main business logic for generating MCQ questions.
"""

import json
import re
from typing import List, Optional
from Domain.entities.mcq_question import MCQQuestion, MCQOption
from Domain.entities.document import Document
from Domain.entities.enums import DifficultyLevel, QuestionType
from Domain.repositories.llm_repository import LLMRepository
from Domain.repositories.vector_store_repository import VectorStoreRepository
from Domain.services.quality_validation_service import QualityValidationService
from Domain.services.difficulty_analysis_service import DifficultyAnalysisService


class MCQGenerationService:
    """Core service for MCQ generation"""

    def __init__(
        self,
        llm_repository: LLMRepository,
        vector_store_repository: VectorStoreRepository,
        quality_service: QualityValidationService,
        difficulty_service: DifficultyAnalysisService
    ):
        self.llm_repository = llm_repository
        self.vector_store_repository = vector_store_repository
        self.quality_service = quality_service
        self.difficulty_service = difficulty_service
        self.max_context_length = 600

    async def generate_mcq(
        self,
        topic: str,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        question_type: QuestionType = QuestionType.DEFINITION,
        context_query: Optional[str] = None
    ) -> MCQQuestion:
        """
        Generate a single MCQ question

        Args:
            topic: The topic for the question
            difficulty: Difficulty level of the question
            question_type: Type of question to generate
            context_query: Optional specific context query

        Returns:
            Generated MCQ question

        Raises:
            ValueError: If no relevant context found or generation fails
        """
        # 1. Retrieve relevant context
        query = context_query or topic
        contexts = await self.vector_store_repository.search_diverse(query, k=3)

        if not contexts:
            raise ValueError(f"No relevant context found for topic: {topic}")

        # 2. Prepare context text
        context_text = self._prepare_context(contexts)

        # 3. Generate MCQ using LLM
        raw_response = await self.llm_repository.generate_mcq(
            context_text, topic, difficulty, question_type
        )

        # 4. Parse and create MCQ object
        mcq = self._parse_mcq_response(raw_response, context_text, topic, difficulty, question_type, contexts[0].source)

        # 5. Calculate quality score
        mcq.confidence_score = self.quality_service.calculate_quality_score(mcq)

        return mcq

    async def generate_batch(
        self,
        topics: List[str],
        questions_per_topic: int = 5,
        difficulties: Optional[List[DifficultyLevel]] = None,
        question_types: Optional[List[QuestionType]] = None
    ) -> List[MCQQuestion]:
        """
        Generate batch of MCQ questions

        Args:
            topics: List of topics to generate questions for
            questions_per_topic: Number of questions per topic
            difficulties: List of difficulty levels to cycle through
            question_types: List of question types to cycle through

        Returns:
            List of generated MCQ questions
        """
        if difficulties is None:
            difficulties = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]

        if question_types is None:
            question_types = [QuestionType.DEFINITION, QuestionType.APPLICATION]

        mcqs = []
        total_questions = len(topics) * questions_per_topic

        for i, topic in enumerate(topics):
            for j in range(questions_per_topic):
                try:
                    # Cycle through difficulties and question types
                    difficulty = difficulties[j % len(difficulties)]
                    question_type = question_types[j % len(question_types)]

                    mcq = await self.generate_mcq(topic, difficulty, question_type)
                    mcqs.append(mcq)

                except Exception as e:
                    print(f"Failed to generate question {j+1} for topic '{topic}': {e}")
                    continue

        return mcqs

    def _prepare_context(self, contexts: List[Document]) -> str:
        """
        Prepare context text from retrieved documents

        Args:
            contexts: List of context documents

        Returns:
            Formatted and truncated context text
        """
        context_text = "\n\n".join(doc.content for doc in contexts)
        return self._truncate_context(context_text)

    def _truncate_context(self, context: str) -> str:
        """
        Intelligently truncate context to fit within token limits

        Args:
            context: Context text to truncate

        Returns:
            Truncated context text
        """
        if len(context) <= self.max_context_length:
            return context

        # Try to truncate at sentence boundary
        sentences = context.split('. ')
        truncated = ""

        for sentence in sentences:
            if len(truncated + sentence + '. ') <= self.max_context_length:
                truncated += sentence + '. '
            else:
                break

        # If no complete sentences fit, truncate at word boundary
        if not truncated:
            words = context.split()
            truncated = ""
            for word in words:
                if len(truncated + word + ' ') <= self.max_context_length:
                    truncated += word + ' '
                else:
                    break

        return truncated.strip()

    def _parse_mcq_response(
        self,
        raw_response: str,
        context: str,
        topic: str,
        difficulty: DifficultyLevel,
        question_type: QuestionType,
        source: str
    ) -> MCQQuestion:
        """
        Parse LLM response into MCQ object

        Args:
            raw_response: Raw response from LLM
            context: Context used for generation
            topic: Question topic
            difficulty: Question difficulty
            question_type: Question type
            source: Source document

        Returns:
            Parsed MCQ question

        Raises:
            ValueError: If parsing fails
        """
        try:
            # Extract JSON from response
            response_data = self._extract_json_from_response(raw_response)

            # Create MCQ options
            options = []
            for label, text in response_data["options"].items():
                is_correct = label == response_data["correct_answer"]
                options.append(MCQOption(label, text, is_correct))

            # Create MCQ object
            mcq = MCQQuestion(
                question=response_data["question"],
                context=context,
                options=options,
                explanation=response_data.get("explanation", ""),
                difficulty=difficulty.value,
                topic=topic,
                question_type=question_type.value,
                source=source
            )

            return mcq

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def _extract_json_from_response(self, response: str) -> dict:
        """
        Extract JSON from LLM response with multiple fallback strategies

        Args:
            response: Raw LLM response

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found
        """
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
