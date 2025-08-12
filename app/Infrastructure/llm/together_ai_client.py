"""
Together AI client implementation.
Implements LLMRepository interface for Together AI API integration.
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional
from Domain.repositories.llm_repository import LLMRepository
from Domain.entities.enums import DifficultyLevel, QuestionType


class TogetherAIClient(LLMRepository):
    """Together AI client implementation"""

    def __init__(
        self,
        api_key: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo",
        base_url: str = "https://api.together.xyz/v1",
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate_mcq(
        self,
        context: str,
        topic: str,
        difficulty: DifficultyLevel,
        question_type: QuestionType
    ) -> str:
        """
        Generate MCQ using Together AI API

        Args:
            context: Relevant context for the question
            topic: The topic for the question
            difficulty: Difficulty level of the question
            question_type: Type of question to generate

        Returns:
            Raw LLM response as string

        Raises:
            Exception: If API call fails after retries
        """
        prompt = self._build_prompt(context, topic, difficulty, question_type)

        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 800,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["Human:", "Assistant:"]
                }

                async with session.post(
                    f"{self.base_url}/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["text"].strip()
                    elif response.status == 429:  # Rate limit
                        wait_time = 2 ** attempt
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        error_text = await response.text()
                        raise Exception(f"Together AI API error {response.status}: {error_text}")

            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    raise Exception("Together AI API timeout after retries")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Together AI API failed after {self.max_retries} retries: {str(e)}")
                await asyncio.sleep(2 ** attempt)

        raise Exception("Unexpected error in Together AI API call")

    async def health_check(self) -> bool:
        """
        Check Together AI service health

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            session = await self._get_session()
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Simple test call
            payload = {
                "model": self.model,
                "prompt": "Test",
                "max_tokens": 1,
                "temperature": 0.1
            }

            async with session.post(
                f"{self.base_url}/completions",
                headers=headers,
                json=payload
            ) as response:
                return response.status == 200

        except Exception:
            return False

    def get_model_info(self) -> dict:
        """
        Get information about the current model

        Returns:
            Dictionary containing model information
        """
        return {
            "provider": "Together AI",
            "model": self.model,
            "base_url": self.base_url,
            "max_retries": self.max_retries,
            "timeout": self.timeout
        }

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for given text

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimation: 1 token ≈ 0.75 words for Vietnamese/English mixed text
        words = len(text.split())
        return int(words * 1.33)

    def _build_prompt(
        self,
        context: str,
        topic: str,
        difficulty: DifficultyLevel,
        question_type: QuestionType
    ) -> str:
        """
        Build prompt for MCQ generation

        Args:
            context: Relevant context
            topic: Question topic
            difficulty: Difficulty level
            question_type: Question type

        Returns:
            Formatted prompt string
        """
        # Get question type instruction
        type_instructions = {
            QuestionType.DEFINITION: "Tạo câu hỏi định nghĩa thuật ngữ hoặc khái niệm.",
            QuestionType.APPLICATION: "Tạo câu hỏi về ứng dụng thực tế của kiến thức.",
            QuestionType.ANALYSIS: "Tạo câu hỏi phân tích hoặc so sánh các khái niệm.",
            QuestionType.COMPARISON: "Tạo câu hỏi so sánh sự khác biệt giữa các khái niệm.",
            QuestionType.EVALUATION: "Tạo câu hỏi đánh giá hoặc nhận xét về phương pháp/kỹ thuật."
        }

        difficulty_instructions = {
            DifficultyLevel.EASY: "Câu hỏi đơn giản, dễ hiểu, phù hợp với người mới bắt đầu.",
            DifficultyLevel.MEDIUM: "Câu hỏi trung bình, cần hiểu biết cơ bản về chủ đề.",
            DifficultyLevel.HARD: "Câu hỏi khó, cần kiến thức sâu và khả năng phân tích.",
            DifficultyLevel.EXPERT: "Câu hỏi chuyên sâu, dành cho người có kinh nghiệm cao."
        }

        type_instruction = type_instructions.get(question_type, type_instructions[QuestionType.DEFINITION])
        difficulty_instruction = difficulty_instructions.get(difficulty, difficulty_instructions[DifficultyLevel.MEDIUM])

        return f"""Hãy tạo 1 câu hỏi trắc nghiệm dựa trên nội dung sau đây.

Nội dung: {context}

Yêu cầu:
- Chủ đề: {topic}
- Mức độ: {difficulty.value} - {difficulty_instruction}
- Loại câu hỏi: {question_type.value} - {type_instruction}

QUAN TRỌNG: Chỉ trả về JSON hợp lệ, không có text bổ sung khác. Luôn trả lời bằng tiếng Việt:

{{
    "question": "Câu hỏi rõ ràng và cụ thể về {topic}",
    "options": {{
        "A": "Đáp án A - chi tiết và rõ ràng",
        "B": "Đáp án B - chi tiết và rõ ràng",
        "C": "Đáp án C - chi tiết và rõ ràng",
        "D": "Đáp án D - chi tiết và rõ ràng"
    }},
    "correct_answer": "A",
    "explanation": "Giải thích chi tiết tại sao đáp án đúng là A, và tại sao các đáp án khác không đúng."
}}"""
