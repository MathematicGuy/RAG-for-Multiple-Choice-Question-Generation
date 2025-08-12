import os
import time
import torch
import json
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.pipelines import pipeline
from transformers.utils.quantization_config import BitsAndBytesConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_core.prompts import PromptTemplate
import textwrap # remove indentation from the JSON block
from langchain.vectorstores import FAISS

#? OOP
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")


def load_llm(model_name):
    token_path = Path("./api_key/hugging_face_token.txt")
    if not token_path.exists():
        raise FileNotFoundError("Missing HuggingFace token.txt")

    with token_path.open("r") as f:
        hf_token = f.read().strip()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map="auto",
        token=hf_token
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=model_pipeline)


def load_documents(folder_path):
    folder = Path(folder_path.strip().strip('"\''))

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files in folder: {folder}")

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


# class State(TypedDict):
#     question: str
#     context: List[Document]
#     answer: str

def build_rag_chain(docs, embeddings, llm):
    chunker = SemanticChunker(
        embeddings=embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    chunks = chunker.split_documents(docs)
    vector_db = FAISS.from_documents(chunks, embedding=embeddings)
    retriever = vector_db.as_retriever(top_k=5, score_threshold=None)

    #? use textwrap to remove indentation from the JSON block
    prompt_text = """
        Bạn là một trợ lý chuyên thực hiện các nhiệm vụ trả lời câu hỏi.
        Hãy sử dụng các phần nội dung được truy xuất bên dưới để trả lời câu hỏi.
        Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết.

        Yêu cầu: Hãy trả về phản hồi dưới dạng một đối tượng JSON hợp lệ, với đúng ba khóa: "context", "question" và "answer". Chỉ xuất ra đối tượng JSON, không thêm bất kỳ nội dung nào khác.

        Ví dụ về đầu ra JSON:
        {{
            "context": "OOP là một mô hình lập trình dựa trên khái niệm đối tượng.",
            "question": "OOP là gì?",
            "answer": "OOP là viết tắt của Lập trình hướng đối tượng, một mô hình tổ chức thiết kế phần mềm xung quanh dữ liệu hoặc đối tượng, thay vì các hàm và logic."
        }}

        Context: {context}
        Question: {question}
        Answer:
    """ #? {} mean Variable. Use {{ }} "escape" the curly braces in your example JSON so that LangChain treats them as literal text,

    prompt_template = PromptTemplate.from_template(prompt_text)


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # rag_chain = (
    #     #? { } -> Syntax for RunnableMap.
    #     #? retriever is the retrieved information. format_docs is function to format the retrieved information.
    #     RunnableParallel({'context': retriever | format_docs, 'question': RunnablePassthrough()})
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # return_json_result = RunnableLambda(lambda result: {
    #     "context": result["context"],
    #     "question": result["question"],
    #     "answer": result
    # })

    #? smart JSON parser help to fix invalid or malformed (i.e not json) outputs from a model automatically by using the LLM itself
    #? basically if the output is broken (e.g. missing commas, extra test) -> fix it using the LLM then parse
    # json_parser = JsonOutputParser()

    rag_chain = (
        RunnableMap({
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        })
        # Takes the context and question and formats them into a single string prompt.
        | prompt_template
        # Receives the formatted prompt and outputs a string, which should be in JSON format.
        | llm
        # Receives the string from the LLM and automatically parses it into a Python dictionary.
        # | json_parser
    )
    return rag_chain, len(chunks)


def main():
    get_device()
    embeddings = load_embeddings()
    #? PhoGPT-5.5B
    #? Phi-2 (2.7B)
    #? lmsys/vicuna-7b-v1.5
    MODEL_NAME= "google/gemma-2b-it"
    # MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    llm = load_llm(MODEL_NAME)

    folder_path = "pdf_folder"  # Replace with your path
    start = time.time()
    docs, filenames = load_documents(folder_path)
    rag_chain, num_chunks = build_rag_chain(docs, embeddings, llm)

    print(f"\nReady: {len(filenames)} files, {num_chunks} chunks")
    print(f"⏱️ Loading Time: {time.time() - start:.2f}s")

    # while True:
    json_outputs = []
    for i in range(5): # Check consistency
        #     user_input = input("\nYour question (type 'exit' to quit): ")
        #     if user_input.lower() == "exit":
        #         break

        #     print("\nGenerating answer...")

        start = time.time()
        # response = rag_chain.invoke('OOP là gì ?')
        response = rag_chain.invoke("OOP là gì ?")
        start = response.rfind("{")
        end = response.rfind("}")
        json_outputs.append(response[start:end+1].strip())

        # try:
        #     print('dsjhfsdjfsdjfhdshfsdkhfkjds--------')
        #     print(f"\nContext: {response['response']['context']}")
        #     print('------------------------------------')
        #     print(f"Questionf: {response['response']['question']}")
        #     print('------------------------------------')
        #     print(f"JSON OUTPUT: {response['response']['answer']}")

        #     with open("output.txt", "a", encoding="utf-8") as f:
        #         f.write(str(response) + "\n")

        # except Exception as error:
        #     print(f"Error: {error}")

        print(f"JSON OUTPUT: {response}")
        print(f"Time taken: {time.time() - start:.2f}s")

    for json_output in json_outputs:
        print(json_output)

    for json_output in json_outputs:
        data = json.loads(json_output)
        print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

    #? Code hoàn thiện bản đầu tiên -> reverse lại về dạng code đơn giản ko có JSON format, FAISS
