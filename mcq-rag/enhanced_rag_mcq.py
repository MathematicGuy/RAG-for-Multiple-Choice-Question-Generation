"""
Enhanced RAG System for Multiple Choice Question Generation
Author: MathematicGuy
Date: July 2025

This module implements an advanced RAG system specifically designed for generating
high-quality Multiple Choice Questions from educational documents.
"""

import time
import numpy as np

# Transformers imports
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from util import debug_prompt_templates
from generator import EnhancedRAGMCQGenerator

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
        folder_path = "./pdfs"  # Updated path to your PDF folder
        try:
            docs, filenames = generator.load_documents(folder_path)
            num_chunks = generator.build_vector_database(docs)
            start = time.time() #? Calc generation time

            print(f"⏱️ Loading Time: {time.time() - start:.2f}s") #? Loading document time
            print(f"📚 System ready with {len(filenames)} files and {num_chunks} chunks")

            # Generate sample MCQs
            topics = ["Object Oriented Programming", "Malware Reverse Engineering", "IoT"]

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
            mcqs = generator.generate_batch(
                topics=topics,
                count_per_topic=n_question
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
