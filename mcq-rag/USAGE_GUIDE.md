# RAG-MCQ System User Manual

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Guide](#installation-guide)
4. [Quick Start Guide](#quick-start-guide)
5. [Configuration](#configuration)
6. [Basic Usage](#basic-usage)
7. [Advanced Features](#advanced-features)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)
11. [FAQ](#faq)
12. [Support](#support)

---

## Overview

The RAG-MCQ (Retrieval-Augmented Generation for Multiple Choice Questions) System is an intelligent educational tool that automatically generates high-quality multiple choice questions from educational documents. The system combines advanced natural language processing, semantic search, and educational best practices to create assessments that are pedagogically sound and contextually relevant.

### Key Features

- 🎯 **Intelligent Question Generation**: Creates MCQs from PDF documents using advanced AI
- 🌐 **Vietnamese Language Support**: Optimized for Vietnamese educational content
- 📚 **Multiple Question Types**: Supports definition, application, analysis, comparison, and evaluation questions
- 📊 **Quality Validation**: Automatic validation and scoring of generated questions
- ⚡ **Batch Processing**: Generate multiple questions efficiently
- 🎓 **Difficulty Assessment**: Intelligent difficulty classification and calibration
- 📈 **Performance Monitoring**: Comprehensive metrics and reporting

### Who Should Use This System

- **Educators**: Teachers and instructors who need to create assessments quickly
- **Educational Institutions**: Schools and universities requiring standardized question banks
- **E-learning Platforms**: Online education providers needing automated assessment generation
- **Content Creators**: Educational content developers and curriculum designers
- **Researchers**: Educational technology researchers studying automated assessment

---

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **CPU**: Dual-core processor (Intel i5 or AMD equivalent)
- **RAM**: 8GB system memory
- **Storage**: 20GB free disk space
- **GPU**: Optional (CPU-only mode available)

#### Recommended Requirements
- **CPU**: Quad-core processor (Intel i7 or AMD equivalent)
- **RAM**: 16GB system memory
- **Storage**: 100GB free disk space (for models and document storage)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (GTX 1080/RTX 3070 or higher)

### Software Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python**: Version 3.8 or higher
- **CUDA**: Version 11.0+ (if using GPU acceleration)

### Network Requirements
- Internet connection for model downloads and API access
- Minimum 10 Mbps for initial setup
- 1 Mbps for regular operation

---

## Installation Guide

### Step 1: Clone the Repository

```bash
git clone https://github.com/MathematicGuy/RAG-for-Multiple-Choice-Question-Generation.git
cd RAG-for-Multiple-Choice-Question-Generation
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv rag_mcq_env

# Activate virtual environment
# Windows
rag_mcq_env\Scripts\activate
# macOS/Linux
source rag_mcq_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install required packages
pip install -r TestingFeatures/generated/requirements.txt

# For GPU support (optional)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Setup API Keys

1. Create the API key directory:
```bash
mkdir -p RAG/api_key
```

2. Add your API keys:
```bash
# Hugging Face token (required for model access)
echo "your_huggingface_token_here" > RAG/api_key/hugging_face_token.txt

# LangSmith API key (optional, for debugging)
echo "your_langsmith_api_key_here" > RAG/api_key/langsmith_api_key.txt
```

### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Start Guide

### Running the Demo Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook TestingFeatures/generated/RAG_MCQ_System_Demo.ipynb
```

2. **Run all cells** to see the system in action with sample documents.

### Basic Command Line Usage

```python
from TestingFeatures.generated.enhanced_rag_mcq import EnhancedRAGMCQGenerator
from TestingFeatures.generated.enhanced_rag_mcq import DifficultyLevel, QuestionType

# Initialize the system
generator = EnhancedRAGMCQGenerator()
generator.initialize_system()

# Load documents
docs, filenames = generator.load_documents("path/to/your/pdf/folder")
generator.build_vector_database(docs)

# Generate a single MCQ
mcq = generator.generate_mcq(
    topic="Object-Oriented Programming",
    difficulty=DifficultyLevel.MEDIUM,
    question_type=QuestionType.DEFINITION
)

# Display the result
print(f"Question: {mcq.question}")
for option in mcq.options:
    marker = "✅" if option.is_correct else "  "
    print(f"  {marker} {option.label}: {option.text}")
print(f"Explanation: {mcq.explanation}")
```

### Quick Document Processing

```python
# For quick testing with sample documents
from pathlib import Path

# Place your PDF files in this folder
pdf_folder = Path("RAG/pdf_folder")
if pdf_folder.exists():
    docs, filenames = generator.load_documents(str(pdf_folder))
    print(f"Loaded {len(docs)} documents from {len(filenames)} files")
else:
    print("Please add PDF files to RAG/pdf_folder/")
```

---

## Configuration

### Configuration File

The system uses `config.json` for configuration. Key settings include:

```json
{
  "system_config": {
    "embedding_model": "bkai-foundation-models/vietnamese-bi-encoder",
    "llm_model": "google/gemma-2b-it",
    "use_quantization": true
  },
  "generation_config": {
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 0.9
  },
  "quality_thresholds": {
    "minimum_score": 60.0,
    "good_score": 75.0,
    "excellent_score": 90.0
  }
}
```

### Environment Variables

Set these environment variables for enhanced functionality:

```bash
# LangSmith integration (optional)
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
export LANGSMITH_API_KEY="your_api_key"
export LANGSMITH_PROJECT="rag-mcq-project"

# Hugging Face
export HUGGING_FACE_API_KEY="your_huggingface_token"
```

### Model Configuration

#### Embedding Models
- **Vietnamese**: `bkai-foundation-models/vietnamese-bi-encoder` (recommended)
- **Multilingual**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **English**: `sentence-transformers/all-MiniLM-L6-v2`

#### Language Models
- **Small**: `google/gemma-2b-it` (2B parameters, fast)
- **Medium**: `lmsys/vicuna-7b-v1.5` (7B parameters, balanced)
- **Large**: `meta-llama/Llama-2-13b-chat-hf` (13B parameters, high quality)

---

## Basic Usage

### 1. Document Management

#### Adding Documents
```python
# Place PDF files in your chosen folder
document_folder = "path/to/your/documents"

# The system will automatically:
# - Extract text from PDFs
# - Create semantic chunks
# - Generate embeddings
# - Build vector database
```

#### Supported Document Types
- **PDF files** (.pdf) - Primary support
- **Text files** (.txt) - Basic support
- **Word documents** (.docx) - Planned support

#### Document Best Practices
- Use well-formatted PDF files with selectable text
- Avoid scanned PDFs (OCR support planned)
- Organize documents by subject for better context retrieval
- Recommended file size: 1-50 MB per PDF

### 2. Question Generation

#### Single Question Generation
```python
# Generate a definition question
mcq = generator.generate_mcq(
    topic="Data Structures",
    difficulty=DifficultyLevel.EASY,
    question_type=QuestionType.DEFINITION
)

# Generate an application question
mcq = generator.generate_mcq(
    topic="Algorithms",
    difficulty=DifficultyLevel.HARD,
    question_type=QuestionType.APPLICATION
)
```

#### Batch Question Generation
```python
# Generate multiple questions
topics = ["OOP", "Data Structures", "Algorithms"]
mcqs = generator.generate_batch(
    topics=topics,
    count_per_topic=5,
    difficulties=[DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD],
    question_types=[QuestionType.DEFINITION, QuestionType.APPLICATION]
)

print(f"Generated {len(mcqs)} questions")
```

### 3. Question Types

#### Definition Questions
- Test understanding of terminology and concepts
- Keywords: "What is...", "Define...", "The meaning of..."
- Best for: Basic comprehension, vocabulary testing

#### Application Questions
- Test ability to apply knowledge in scenarios
- Keywords: "When to use...", "In which case...", "Apply..."
- Best for: Practical understanding, real-world application

#### Comparison Questions
- Test understanding of similarities and differences
- Keywords: "Compare...", "Difference between...", "Similar to..."
- Best for: Conceptual relationships, critical thinking

#### Analysis Questions
- Test ability to break down complex information
- Keywords: "Analyze...", "What causes...", "Why does..."
- Best for: Deep understanding, problem-solving

#### Evaluation Questions
- Test ability to make judgments and decisions
- Keywords: "Best approach...", "Most effective...", "Evaluate..."
- Best for: Critical thinking, decision making

### 4. Difficulty Levels

#### Easy (Basic Level)
- Simple recall and recognition
- Basic definitions and examples
- Single concept focus
- Target: Beginner learners

#### Medium (Intermediate Level)
- Comprehension and application
- Multiple concept integration
- Moderate cognitive load
- Target: Regular students

#### Hard (Advanced Level)
- Analysis and evaluation
- Complex scenarios
- High cognitive load
- Target: Advanced students

#### Expert (Professional Level)
- Synthesis and creation
- Professional scenarios
- Maximum cognitive complexity
- Target: Graduate students, professionals

---

## Advanced Features

### 1. Custom Prompt Engineering

#### Creating Custom Prompts
```python
from TestingFeatures.generated.enhanced_rag_mcq import AdvancedPromptManager

prompt_manager = AdvancedPromptManager()

# Create custom prompt for specific domain
custom_prompt = """
Create a programming question focused on {topic}.
Include code examples and practical scenarios.
Difficulty: {difficulty}
Context: {context}
"""

# Use custom prompt
mcq = generator.generate_mcq_with_custom_prompt(
    context="Your context here",
    prompt_template=custom_prompt,
    topic="Python Programming",
    difficulty=DifficultyLevel.MEDIUM
)
```

#### Prompt Templates by Domain
- **Computer Science**: Code analysis, algorithm questions
- **Mathematics**: Problem-solving, theorem application
- **Science**: Experimental analysis, concept application
- **Language Arts**: Reading comprehension, grammar

### 2. Quality Control

#### Automatic Validation
```python
from TestingFeatures.generated.enhanced_rag_mcq import QualityValidator

validator = QualityValidator()

# Validate a single question
is_valid, validation_results = validator.validate_mcq(mcq)
confidence_score = validator.calculate_confidence_score(mcq)

print(f"Valid: {is_valid}")
print(f"Quality Score: {confidence_score}/100")
print(f"Issues: {validation_results['issues']}")
```

#### Custom Quality Criteria
```python
# Set custom quality thresholds
validator.min_question_length = 15
validator.max_question_length = 150
validator.min_explanation_length = 30

# Generate quality report
mcqs = [mcq1, mcq2, mcq3]  # Your generated MCQs
quality_report = validator.generate_quality_report(mcqs)
print(f"Average quality: {quality_report['average_score']}")
```

### 3. Performance Optimization

#### GPU Acceleration
```python
# Check GPU availability
import torch
if torch.cuda.is_available():
    config["system_config"]["device"] = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    config["system_config"]["device"] = "cpu"
    print("Using CPU")
```

#### Memory Optimization
```python
# Enable quantization for memory efficiency
config["system_config"]["use_quantization"] = True

# Adjust batch size based on available memory
if torch.cuda.is_available():
    config["performance"]["batch_size"] = 10
else:
    config["performance"]["batch_size"] = 5
```

#### Caching
```python
# Enable embedding caching
config["performance"]["cache_embeddings"] = True

# Enable response caching
config["performance"]["cache_responses"] = True
```

### 4. Export and Integration

#### Export Formats
```python
# Export to JSON
generator.export_mcqs(mcqs, "questions.json")

# Export to CSV
import pandas as pd
df = pd.DataFrame([mcq.to_dict() for mcq in mcqs])
df.to_csv("questions.csv", index=False)

# Export to XML
import xml.etree.ElementTree as ET
# Custom XML export implementation
```

#### Learning Management System Integration
- **Moodle**: XML import format
- **Canvas**: QTI format support
- **Blackboard**: Custom format conversion
- **Google Classroom**: Direct integration via API

---

## API Reference

### Core Classes

#### EnhancedRAGMCQGenerator
Main class for the RAG-MCQ system.

```python
class EnhancedRAGMCQGenerator:
    def __init__(self, config: Optional[Dict[str, Any]] = None)
    def initialize_system(self) -> None
    def load_documents(self, folder_path: str) -> Tuple[List[Document], List[str]]
    def build_vector_database(self, docs: List[Document]) -> int
    def generate_mcq(self, topic: str, difficulty: DifficultyLevel,
                     question_type: QuestionType) -> MCQQuestion
    def generate_batch(self, topics: List[str], count_per_topic: int = 5) -> List[MCQQuestion]
    def export_mcqs(self, mcqs: List[MCQQuestion], output_path: str) -> None
```

#### MCQQuestion
Data class representing a multiple choice question.

```python
@dataclass
class MCQQuestion:
    question: str
    context: str
    options: List[MCQOption]
    explanation: str
    difficulty: str
    topic: str
    question_type: str
    source: str
    confidence_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]
```

#### QualityValidator
Validates and scores generated MCQs.

```python
class QualityValidator:
    def validate_mcq(self, mcq: MCQQuestion) -> Tuple[bool, Dict[str, Any]]
    def calculate_confidence_score(self, mcq: MCQQuestion) -> float
    def generate_quality_report(self, mcqs: List[MCQQuestion]) -> Dict[str, Any]
```

### Enumerations

#### DifficultyLevel
```python
class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
```

#### QuestionType
```python
class QuestionType(Enum):
    DEFINITION = "definition"
    COMPARISON = "comparison"
    APPLICATION = "application"
    ANALYSIS = "analysis"
    EVALUATION = "evaluation"
```

### Configuration Options

#### System Configuration
- `embedding_model`: Model for text embeddings
- `llm_model`: Language model for generation
- `device`: Processing device (cpu/cuda)
- `use_quantization`: Enable model quantization

#### Generation Configuration
- `temperature`: Randomness in generation (0.0-1.0)
- `max_tokens`: Maximum output length
- `top_p`: Nucleus sampling parameter
- `repetition_penalty`: Penalty for repetition

#### Quality Configuration
- `minimum_score`: Minimum quality threshold
- `good_score`: Good quality threshold
- `excellent_score`: Excellence threshold

---

## Troubleshooting

### Common Issues and Solutions

#### Installation Issues

**Problem**: Package installation fails
```bash
# Solution: Upgrade pip and try again
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**Problem**: CUDA not detected
```bash
# Check CUDA installation
nvidia-smi
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Model Loading Issues

**Problem**: Hugging Face model download fails
```python
# Solution: Check API token and internet connection
import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "/path/to/cache"
# Verify token in RAG/api_key/hugging_face_token.txt
```

**Problem**: Out of memory errors
```python
# Solution: Reduce model size or enable quantization
config["system_config"]["use_quantization"] = True
config["system_config"]["llm_model"] = "google/gemma-2b-it"  # Smaller model
```

#### Generation Issues

**Problem**: Low quality questions generated
```python
# Solutions:
# 1. Improve document quality and formatting
# 2. Adjust prompt engineering
# 3. Increase quality thresholds
# 4. Use larger language models
```

**Problem**: Questions not relevant to documents
```python
# Solutions:
# 1. Check document loading and chunking
# 2. Verify embedding model language compatibility
# 3. Adjust retrieval parameters
# 4. Improve topic specification
```

#### Performance Issues

**Problem**: Slow generation speed
```python
# Solutions:
# 1. Enable GPU acceleration
# 2. Use quantized models
# 3. Reduce batch size
# 4. Enable caching
```

**Problem**: High memory usage
```python
# Solutions:
# 1. Use smaller models
# 2. Enable quantization
# 3. Process documents in smaller batches
# 4. Clear cache regularly
```

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Set environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "rag-mcq-debug"
```

### Log Files

Check these log files for troubleshooting:
- `rag_mcq.log`: System operation logs
- `generation.log`: Question generation logs
- `validation.log`: Quality validation logs

---

## Best Practices

### Document Preparation

1. **Use High-Quality PDFs**
   - Ensure text is selectable (not scanned images)
   - Use well-formatted documents with clear structure
   - Include headers, subheadings, and proper formatting

2. **Organize Content**
   - Group related documents by subject
   - Use descriptive filenames
   - Maintain consistent formatting across documents

3. **Content Guidelines**
   - Include diverse content types (definitions, examples, procedures)
   - Ensure factual accuracy
   - Use clear, educational language

### Question Generation

1. **Topic Selection**
   - Use specific, focused topics
   - Avoid overly broad or vague topics
   - Align topics with learning objectives

2. **Difficulty Progression**
   - Start with easier questions to build confidence
   - Gradually increase difficulty
   - Balance different difficulty levels

3. **Question Type Variety**
   - Mix different question types
   - Match question types to learning goals
   - Consider cognitive load for target audience

### Quality Assurance

1. **Regular Validation**
   - Review generated questions manually
   - Use quality metrics consistently
   - Get feedback from subject matter experts

2. **Continuous Improvement**
   - Monitor generation performance
   - Update prompts based on results
   - Retrain models with better data

3. **Content Review**
   - Check for bias and fairness
   - Ensure cultural appropriateness
   - Verify factual accuracy

### Performance Optimization

1. **Resource Management**
   - Monitor system resources
   - Use appropriate hardware
   - Scale based on usage patterns

2. **Batch Processing**
   - Process documents in optimal batch sizes
   - Use parallel processing when available
   - Cache frequently used embeddings

3. **Model Selection**
   - Choose models appropriate for your use case
   - Balance quality and performance
   - Consider language-specific models

---

## FAQ

### General Questions

**Q: What types of documents can the system process?**
A: Currently, the system primarily supports PDF documents with selectable text. Text files (.txt) are also supported. Future versions will include support for Word documents and scanned PDFs with OCR.

**Q: How many questions can the system generate at once?**
A: The system can generate hundreds of questions in batch mode. Performance depends on your hardware and the complexity of source documents. Typical batch sizes range from 10-100 questions.

**Q: Can I use the system for languages other than Vietnamese?**
A: Yes, the system supports multiple languages. You can configure different embedding models for other languages (English, Chinese, etc.) by modifying the configuration file.

**Q: How accurate are the generated questions?**
A: Question accuracy depends on source document quality and system configuration. With high-quality documents and proper configuration, the system typically achieves 80-95% accuracy in educational assessments.

### Technical Questions

**Q: What hardware do I need to run the system?**
A: Minimum: 8GB RAM, dual-core CPU. Recommended: 16GB RAM, GPU with 8GB VRAM, quad-core CPU. See the [System Requirements](#system-requirements) section for details.

**Q: Can I run the system on a cloud platform?**
A: Yes, the system works on cloud platforms like AWS, Google Cloud, and Azure. Use GPU instances for better performance.

**Q: How do I integrate the system with my existing LMS?**
A: The system can export questions in various formats (JSON, CSV, XML). Most LMS platforms support question import through QTI or custom formats.

**Q: Can I customize the question generation prompts?**
A: Yes, the system includes an advanced prompt management system. You can create custom prompts for specific domains or question types.

### Usage Questions

**Q: How do I improve question quality?**
A: 1) Use high-quality, well-formatted documents, 2) Configure appropriate quality thresholds, 3) Use larger language models, 4) Customize prompts for your domain.

**Q: What if the system generates duplicate questions?**
A: The system includes duplicate detection. You can also increase the diversity threshold in the configuration to reduce similar questions.

**Q: How do I handle different difficulty levels?**
A: The system automatically assesses and assigns difficulty levels. You can also specify target difficulty levels when generating questions.

**Q: Can I review and edit generated questions?**
A: Yes, all questions are exportable for manual review and editing. The system also provides quality scores to help prioritize review efforts.

---

## Support

### Getting Help

#### Documentation
- **User Manual**: This document
- **API Documentation**: In-code documentation and examples
- **Technical Guide**: `Instruction.md` for detailed technical information

#### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share experiences
- **Wiki**: Community-contributed guides and tips

#### Professional Support
For professional support, training, or custom development:
- Email: [Contact information]
- Website: [Project website]
- Training: Available for institutional users

### Reporting Issues

When reporting issues, please include:
1. **System Information**: OS, Python version, hardware specs
2. **Error Messages**: Complete error logs and stack traces
3. **Reproduction Steps**: Clear steps to reproduce the issue
4. **Configuration**: Your config.json and environment settings
5. **Sample Data**: If possible, provide sample documents (without sensitive content)

### Feature Requests

We welcome feature requests! Please:
1. Check existing issues first
2. Provide clear use case descriptions
3. Include mockups or examples if applicable
4. Consider contributing to development

### Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Follow coding standards
4. Add tests for new features
5. Submit a pull request

### Roadmap

Upcoming features:
- **Multi-modal support**: Images and diagrams
- **Advanced analytics**: Learning outcome tracking
- **API development**: REST API for integration
- **Mobile support**: Mobile app development
- **Collaborative features**: Multi-user question review

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- LangChain community for the RAG framework
- Hugging Face for transformer models
- BKAI Foundation for Vietnamese language models
- Open source community for various libraries and tools

---

*Last updated: July 23, 2025*
*Version: 1.0.0*

For the latest updates and documentation, visit the project repository.