from util import QuestionType
from typing import Dict

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
