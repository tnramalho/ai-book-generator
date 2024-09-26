import asyncio
import hashlib
import logging
import os
import re
import json
from typing import List, Dict, Optional, Tuple, Union

import gradio as gr
import tiktoken
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import PromptHelper
from llama_index.core import PromptTemplate

from llama_index.core.retrievers import VectorIndexRetriever
# from llama_index.tools.summarization import HuggingFaceSummarization
from transformers import pipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
# from llama_index.core.prompts import FewShotPromptTemplate


from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
# from llama_index.llms import OpenAI
from llama_index.core.node_parser import SimpleFileNodeParser

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.workflow import step
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, Field, ValidationError
from textstat import textstat
from tortoise import Tortoise, fields, models, run_async
from tortoise.contrib.pydantic import pydantic_model_creator

# --- Configuration and Constants ---
class Config:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite://:memory:"
    )  # Using in-memory SQLite for testing
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")  # Make sure to set your API key
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"  # Sticking with GPT-4o-mini
    DEFAULT_LANGUAGE: str = "Portuguese"
    CONTEXT_WINDOW: int = 4096
    LANCEDB_URI: str = "./lancedb"
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    MAX_MEMORY_WORDS: int = 500  # Adjusted for GPT-4o-mini
    MAX_MEMORY_SENTENCES: int = 20  # Adjusted for GPT-4o-mini
    MAX_MEMORY_TOKENS: int = 4096  # Adjusted for GPT-4o-mini
    MAX_RETRIES: int = 3
    CHAT_MEMORY_TOKEN_LIMIT: int = 512  # Adjusted for GPT-4o-mini
    CONCURRENCY_COUNT: int = 5

config = Config()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Language Prompts ---
LANGUAGE_PROMPTS = {
    "Portuguese": {
        "init_prompt": """
Por favor, escreva um livro técnico abrangente sobre {book_type} sobre {description} com 20 capítulos. 
Siga o formato abaixo com precisão:

* **Título:** <O título do livro>
* **Índice:** <Lista de capítulos com seus títulos>
* **Capítulo 1: Introdução à {book_type}**
* **Capítulo 2: Conceitos Chave em {book_type}**
* **Capítulo 3: Tópicos Avançados em {book_type}**
...
* **Resumo:** <Um breve resumo dos três primeiros capítulos>
* **Instrução 1:** <Uma instrução sobre o que deve ser abordado no Capítulo 4>
* **Instrução 2:** <Outra instrução para um possível tópico de capítulo>
* **Instrução 3:** <Uma terceira instrução para uma direção potencial do capítulo>

Certifique-se de ser preciso e seguir estritamente o formato de saída.
""",
        "human_prompt": """
Agora imagine que você é um assistente útil que ajuda um autor a estruturar um livro técnico sobre IA. 
Você receberá um tópico atual, um resumo do tópico e 3 planos diferentes para desenvolver o próximo tópico.

Preciso que você:
1. Selecione o plano mais relevante e abrangente proposto.
2. Explique por que escolheu esse plano.
3. Revise o plano selecionado em um esboço detalhado para o próximo tópico.

Tópico Atual:  
{current_topic}

O resumo do tópico atual:
{summary}

Três planos para o próximo tópico:
{plans}

Agora comece escolhendo e revisando, organizando sua saída seguindo estritamente o seguinte formato:

Plano Selecionado: 
<copia o plano selecionado aqui>
Razão:
<Explique por que escolheu o plan>
Plano Revisado:
<string do plano revisado>, mantenha detalhado, cerca de 5-7 frases.
""",
        "chapter_section_prompt": """
### {section_title}

{section_content}

**Perguntas e Respostas:**

* Pergunta 1: {question1}
* Resposta 1: {answer1}
* Pergunta 2: {question2}
* Resposta 2: {answer2}
""",
        "chapter_prompt": """
## Capítulo {chapter_number}: {chapter_title}

{chapter_sections}
""",
    },
    # Add prompts for other languages here
}

# --- Database Models ---
class Book(models.Model):
    id = fields.IntField(pk=True)
    title = fields.CharField(max_length=255, null=False)
    table_of_contents = fields.TextField(null=False)
    language = fields.CharField(max_length=50, null=False)
    model = fields.CharField(max_length=50, null=False)

    class Meta:
        table = "books"


class Chapter(models.Model):
    id = fields.IntField(pk=True)
    book = fields.ForeignKeyField("models.Book", related_name="chapters")
    chapter_number = fields.IntField(null=False)
    title = fields.CharField(max_length=255, null=False)
    content = fields.TextField(null=False)

    class Meta:
        table = "chapters"


# --- Pydantic Models ---
BookSchema = pydantic_model_creator(Book, name="Book", exclude_readonly=True)
ChapterSchema = pydantic_model_creator(Chapter, name="Chapter", exclude_readonly=True)


class BookOutputInstruction(BaseModel):
    Instruction_1: str = Field(
        ..., min_length=5, description="A possible interesting module for the topic"
    )
    Instruction_2: str = Field(
        ..., min_length=5, description="Another possible module direction"
    )
    Instruction_3: str = Field(
        ..., min_length=5, description="Yet another option for module development"
    )


class BookOutputMemory(BaseModel):
    rational: str = Field(..., description="Explanation of memory updates")
    updated_memory: str = Field(
        ...,
        description="Rewritten memory summary",
        max_length=config.MAX_MEMORY_WORDS,
    )


class BookOutput(BaseModel):
    output_modules: Dict[str, str] = Field(
        ..., description="Modules covering key concepts"
    )
    output_summary: str = Field(
        ..., min_length=20, description="Summary of the topic"
    )
    output_questions_answers: Dict[str, str] = Field(
        ..., description="Questions and answers for reinforcement"
    )
    output_instruction: BookOutputInstruction
    output_memory: BookOutputMemory


# --- Utility Functions ---
def get_content_between_a_b(a: str, b: str, text: str) -> str:
    """Extracts content between two substrings."""
    pattern = re.compile(f"{re.escape(a)}(.*?){re.escape(b)}", re.DOTALL)
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def hash_cookie(cookie: str) -> str:
    """Hashes the cookie."""
    return hashlib.sha256(cookie.encode()).hexdigest()


# --- Readability Metrics ---
def pre_process_text(text: str) -> str:
    """Preprocesses text for readability calculations."""
    text = re.sub(r"\n\s*\n", "\n\n", text.strip())
    sections = [
        section.strip() for section in re.split(r"\n{2,}", text) if section.strip()
    ]
    return " ".join(sections)


def automated_readability_index(text: str) -> float:
    """Calculates the Automated Readability Index (ARI)."""
    processed_text = pre_process_text(text)
    return textstat.automated_readability_index(processed_text)


def flesch_reading_ease(text: str) -> float:
    """Calculates the Flesch Reading Ease score."""
    processed_text = pre_process_text(text)
    return textstat.flesch_reading_ease(processed_text)


def gunning_fog_index(text: str) -> float:
    """Calculates the Gunning Fog Index."""
    processed_text = pre_process_text(text)
    return textstat.gunning_fog(processed_text)


def categorize_metrics(
    ari: float, flesch: float, gunning_fog: float
) -> Tuple[str, str, str]:
    """Categorizes readability metrics."""
    clarity_category = (
        "Low" if ari <= 5.0 else "Normal" if ari <= 10.0 else "High"
    )
    understandability_category = (
        "Low" if flesch <= 29.0 else "Normal" if flesch <= 69.0 else "High"
    )
    conciseness_category = (
        "Low"
        if gunning_fog <= 6.0
        else "Normal"
        if gunning_fog <= 12.0
        else "High"
    )
    return clarity_category, understandability_category, conciseness_category


def calculate_readability_metrics(
    text: str,
) -> Tuple[str, float, float, float, str, str, str]:
    """Calculates and categorizes readability metrics."""
    ari = automated_readability_index(text)
    flesch = flesch_reading_ease(text)
    gunning_fog = gunning_fog_index(text)
    clarity, understandability, conciseness = categorize_metrics(
        ari, flesch, gunning_fog
    )
    return clarity, flesch, gunning_fog, ari, understandability, conciseness


# --- Session Management ---
_CACHE = {}

# --- Custom Workflow Events ---
class StoryGenerationDone(Event):
    output: str
    input_data: dict


class OutputValidationError(Event):
    error: str
    wrong_output: str
    input_data: dict


# --- Workflows ---
class RecurrentGPTWorkflow(Workflow):
    def __init__(
        self,
        model_name: str = config.DEFAULT_LLM_MODEL,
        language: str = config.DEFAULT_LANGUAGE,
    ):
        super().__init__(timeout=120, verbose=True)
        self.model_name = model_name
        self.language = language
        self.llm = OpenAI(
            api_key=config.OPENAI_API_KEY, temperature=0, model_name=self.model_name
        )
        self.embedder = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
        self.vector_store = LanceDBVectorStore(
            uri=config.LANCEDB_URI, mode="append"
        )
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            prompt_helper=PromptHelper(context_window=config.CONTEXT_WINDOW),
            embed_model=self.embedder,
        )
        self.long_term_memory_index = VectorStoreIndex(
            [], service_context=self.service_context, vector_store=self.vector_store
        )
        self.retriever = VectorIndexRetriever(
            index=self.long_term_memory_index, similarity_top_k=5
        )
        self.query_engine = RetrieverQueryEngine(retriever=self.retriever)
        self.chat_memory = ChatSummaryMemoryBuffer.from_defaults(
            llm=self.llm,
            token_limit=config.CHAT_MEMORY_TOKEN_LIMIT,
            tokenizer_fn=tiktoken.encoding_for_model(self.model_name).encode,
        )
        self.long_memory: List[str] = []
        self.short_memory: str = ""
        self.output: Dict = {}
        self.node_parser = SimpleFileNodeParser()
        self.summarizer = pipeline(
            model_name=config.SUMMARIZATION_MODEL,
            service_context=self.service_context,
        )
        self.tokenizer = tiktoken.encoding_for_model(self.model_name).encode

        self.few_shot_book_prompt = self.get_story_prompt_template()

    def get_story_prompt_template(self) -> PromptTemplate:
        language = self.language
        prompt_templates = LANGUAGE_PROMPTS

        if language not in prompt_templates:
            logger.warning(
                f"Unsupported language '{language}'. Defaulting to English."
            )
            language = "English"

        template = prompt_templates[language]["init_prompt"] + """
## Technical Book Writing Assistant

**Your Task:** Develop the next section of our technical book on {book_type}. Consider the previous topic, the current memory, and the instructions for what should be covered next.

**Context:**
* Current Topic: {input_topic}
* Current Memory (Summary of the Book So Far): {current_memory}
* Related Past Topics: {related_topics}
* Instructions for What Should Happen Next: {input_instruction}
* New Concept Prompt: {new_concept_prompt}
* Relevant Memories: {relevant_memories}

**Output Format:**
```json
{{
    "output_modules": {{
      "Module_1": "<First key concept>",
      "Module_2": "<Second key concept>",
      "Module_3": "<Third key concept>"
    }},
    "output_summary": "<Summary of the topic>",
    "output_questions_answers": {{
      "Q1": "<First question>",
      "A1": "<Answer to first question>",
      "Q2": "<Second question>",
      "A2": "<Answer to second question>"
    }},
    "output_instruction": {{
      "Instruction_1": "<A possible interesting module for the next topic>",
      "Instruction_2": "<Another possible module direction>", 
      "Instruction_3": "<Yet another option for module development>" 
    }},
    "output_memory": {{
      "rational": "<Explanation of memory updates>",
      "updated_memory": "<Updated summary of the book so far>"
    }}
}}
```

Important:
Keep the Updated Memory concise, around 10-20 sentences.
Never exceed {max_memory_words} words in the Updated Memory.
Ensure technical accuracy and clarity. The content should be informative and educational.
"""

        return PromptTemplate(
            template=template,
            input_variables=[
                "book_type",
                "input_topic",
                "current_memory",
                "related_topics",
                "input_instruction",
                "new_concept_prompt",
                "relevant_memories",
                "max_memory_words"
            ]
        )

    @step
    async def generate_book_section(
        self, ev: Union[StartEvent, OutputValidationError]
    ) -> Union[StoryGenerationDone, StopEvent]:
        """Generates the next section of the technical book."""
        current_retries = await self.ctx.get("retries", default=0)
        if current_retries >= config.MAX_RETRIES:
            logger.warning("Max retries reached.")
            return StopEvent(result="Max retries reached")
        await self.ctx.set("retries", current_retries + 1)
        if isinstance(ev, StartEvent):
            input_data = ev.get("input_data")
            if not input_data:
                logger.error("No input data provided.")
                return StopEvent(result="Please provide input data")
            reflection_prompt = ""
        elif isinstance(ev, OutputValidationError):
            input_data = ev.input_data
            reflection_prompt = (
                f"The previous output was not in the correct format. "
                f"Here's the error: {ev.error}\n"
                f"Please try again, ensuring the output strictly follows the specified format."
            )
        else:
            logger.error("Unknown event type.")
            return StopEvent(result="Unknown event type")

        # Prepare the prompt
        prompt = self.few_shot_book_prompt.format(**input_data) + reflection_prompt
        logger.info(f"Generating book section with prompt: {prompt[:500]}...")

        try:
            response = await self.llm.acomplete(prompt)
            logger.info("Book section generated successfully.")
        except Exception as e:
            logger.error(f"Error during LLM query: {e}")
            return StopEvent(result="Error generating book section.")

        return StoryGenerationDone(output=response, input_data=input_data)

    @step
    async def validate_output(
        self, ev: StoryGenerationDone
    ) -> Union[StopEvent, OutputValidationError]:
        """Validates the LLM output."""
        try:
            # Parse the output
            parsed_output = BookOutput.parse_raw(ev.output)
            logger.info("Output parsed successfully.")
            # Enforce memory limits
            parsed_output.output_memory.updated_memory = (
                self._manage_memory_limits(
                    parsed_output.output_memory.updated_memory
                )
            )
            logger.info("Memory limits enforced.")

            # Update long-term memory
            self.long_memory.append(parsed_output.output_summary)
            # STEP 1: Summarize the output summary
            summary = self.summarizer.summarize(parsed_output.output_summary)
            
            # STEP 2: Insert the summary into the long-term memory index
            self.long_term_memory_index.insert(summary)
            logger.info("Long-term memory updated.")

            # Update short-term memory
            self.short_memory = parsed_output.output_memory.updated_memory
            self.chat_memory.put(
                ChatMessage(
                    role=MessageRole.USER,
                    content=parsed_output.output_instruction.Instruction_1,
                )
            )
            self.chat_memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT, content=parsed_output.output_summary
                )
            )
            logger.info("Short-term memory updated.")

            self.output = parsed_output.dict()

            return StopEvent(result=self.output)
        except ValidationError as e:
            logger.error(f"Validation failed: {e}")
            return OutputValidationError(
                error=str(e), wrong_output=ev.output, input_data=ev.input_data
            )
        except Exception as e:
            logger.error(f"Unexpected error during validation: {e}")
            return OutputValidationError(
                error=str(e), wrong_output=ev.output, input_data=ev.input_data
            )

    def _manage_memory_limits(self, new_memory: str) -> str:
        """Enforces memory size limits."""
        # Enforce sentence and word limits
        sentences = new_memory.split(".")
        if len(sentences) > config.MAX_MEMORY_SENTENCES:
            new_memory = ".".join(sentences[: config.MAX_MEMORY_SENTENCES])
            logger.debug("Memory truncated based on sentence limit.")
        words = new_memory.split()
        if len(words) > config.MAX_MEMORY_WORDS:
            new_memory = " ".join(words[: config.MAX_MEMORY_WORDS])
            logger.debug("Memory truncated based on word limit.")

        # Enforce token limits
        tokens = self.tokenizer(new_memory)
        if len(tokens) > config.MAX_MEMORY_TOKENS:
            new_memory = self.tokenizer.decode(
                tokens[: config.MAX_MEMORY_TOKENS]
            )  # Fixed: Use self.tokenizer.decode
            logger.debug("Memory truncated based on token limit.")

        return new_memory

    async def run_step(self, input_data: Dict) -> Dict:
        """Runs the workflow."""
        start_event = StartEvent(input_data=input_data)
        result_event = await self.arun(start_event)
        return result_event.result if isinstance(result_event, StopEvent) else {}


class HumanWorkflow(Workflow):
    def __init__(
        self,
        model_name: str = config.DEFAULT_LLM_MODEL,
        language: str = config.DEFAULT_LANGUAGE,
    ):
        super().__init__(timeout=120, verbose=True)
        self.model_name = model_name
        self.language = language
        self.llm = OpenAI(
            api_key=config.OPENAI_API_KEY, temperature=0, model_name=self.model_name
        )

    @step
    async def select_and_revise_plan(
        self, ev: StartEvent
    ) -> Union[StopEvent, OutputValidationError]:
        """Allows human to select and revise a plan."""
        input_data = ev.get("input_data")  # Consistent use of get method
        if not input_data:
            logger.error("No input data provided for human workflow.")
            return StopEvent(result="No input data provided")

        prompt = self.prepare_prompt(input_data)
        logger.info(f"Human workflow prompt: {prompt[:500]}...")

        try:
            response = await self.llm.acomplete(prompt)
            logger.info("Human workflow response received.")
        except Exception as e:
            logger.error(f"Error during human workflow LLM query: {e}")
            return OutputValidationError(
                error="Error during human workflow LLM query.",
                wrong_output="",
                input_data=input_data,
            )

        revised_plan = self.parse_revised_plan(response)

        if not revised_plan:
            logger.error("Revised plan not found in response.")
            return OutputValidationError(
                error="Revised plan not found in response.",
                wrong_output=response,
                input_data=input_data,
            )

        output = {"output_instruction": revised_plan}

        logger.info("Revised plan parsed successfully.")
        return StopEvent(result=output)

    def prepare_prompt(self, input_data: Dict) -> str:
        """Prepares the prompt for human interaction."""
        language = self.language
        prompt_templates = LANGUAGE_PROMPTS

        if language not in prompt_templates:
            logger.warning(
                f"Unsupported language '{language}'. Defaulting to English."
            )
            language = "English"

        plans = input_data.get("output_instruction", {})
        plans_list = [
            plans.get("Instruction_1", ""),
            plans.get("Instruction_2", ""),
            plans.get("Instruction_3", ""),
        ]
        plans_formatted = "\n".join(
            [f"{i+1}. {plan}" for i, plan in enumerate(plans_list) if plan]
        )

        prompt = prompt_templates[language]["human_prompt"].format(
            current_topic=input_data.get("input_topic", ""),
            summary=input_data.get("output_summary", ""),
            plans=plans_formatted,
        )
        return prompt

    def parse_revised_plan(self, response: str) -> Optional[str]:
        """Parses the revised plan from the human's response."""
        language = self.language
        if language == "English":
            revised_plan = get_content_between_a_b("Revised Plan:", "", response)
        elif language == "Spanish":
            revised_plan = get_content_between_a_b(
                "Plan Revisado:", "", response
            )
        elif language == "Portuguese":
            revised_plan = get_content_between_a_b(
                "Plano Revisado:", "", response
            )
        else:
            revised_plan = get_content_between_a_b("Revised Plan:", "", response)
        return revised_plan if revised_plan else None

    async def run_step(self, input_data: Dict) -> Optional[str]:
        """Runs the human interaction workflow."""
        start_event = StartEvent(input_data=input_data)
        result_event = await self.arun(start_event)
        return (
            result_event.result.get("output_instruction")
            if isinstance(result_event, StopEvent)
            else None
        )


# --- Gradio Interface Functions ---
def init_prompt(book_type: str, description: str, language: str) -> str:
    """Creates the initial prompt for book generation."""
    language = language if language in LANGUAGE_PROMPTS else "English"
    prompt_template = LANGUAGE_PROMPTS[language]["init_prompt"]
    return prompt_template.format(book_type=book_type, description=description)


async def generate_chapter(
    book: Book,
    chapter_number: int,
    input_data: Dict,
    recurrent_workflow: RecurrentGPTWorkflow,
) -> None:
    """Generates and saves a chapter to the database."""
    recurrent_output = await recurrent_workflow.run_step(input_data)
    logger.info(f"Recurrent Workflow Output: {recurrent_output}")
    chapter_title = recurrent_output["output_modules"]["Module_1"]
    chapter_sections = []

    # Generate sections within the chapter
    for i in range(1, 4):  # Generate 3 sections per chapter
        section_title = recurrent_output["output_modules"][f"Module_{i}"]
        section_content_initial = recurrent_output["output_summary"]

        # Generate questions and answers for each section
        qa_prompt = f"Generate two insightful questions and their comprehensive answers based on the following section content:\n\n{section_content_initial}"
        qa_response = await recurrent_workflow.llm.acomplete(qa_prompt)
        logger.info(f"Q&A Response: {qa_response}")

        try:
            qa_json = json.loads(qa_response)
            question1 = qa_json.get("Q1", "")
            answer1 = qa_json.get("A1", "")
            question2 = qa_json.get("Q2", "")
            answer2 = qa_json.get("A2", "")
        except json.JSONDecodeError:
            logger.error(f"Error decoding Q&A response as JSON: {qa_response}")
            question1 = ""
            answer1 = ""
            question2 = ""
            answer2 = ""

        # Generate the section content with Q&A
        section_content = LANGUAGE_PROMPTS["Portuguese"][
            "chapter_section_prompt"
        ].format(
            section_title=section_title,
            section_content=section_content_initial,
            question1=question1,
            answer1=answer1,
            question2=question2,
            answer2=answer2,
        )
        chapter_sections.append(section_content)

    # Combine sections into the final chapter content
    chapter_content = LANGUAGE_PROMPTS["Portuguese"]["chapter_prompt"].format(
        chapter_number=chapter_number,
        chapter_title=chapter_title,
        chapter_sections="\n".join(chapter_sections),
    )

    chapter = await Chapter.create(
        book=book,
        chapter_number=chapter_number,
        title=chapter_title,
        content=chapter_content,
    )
    logger.info(f"Chapter {chapter.chapter_number} saved: {chapter.title}")


# --- Gradio Functions ---
def on_select_plan(
    selected_plan: str,
    instruction1: str,
    instruction2: str,
    instruction3: str,
) -> str:
    """Updates the selected instruction based on the radio button choice."""
    if selected_plan == "Instruction 1":
        return instruction1
    elif selected_plan == "Instruction 2":
        return instruction2
    elif selected_plan == "Instruction 3":
        return instruction3
    else:
        return ""

# --- Gradio App ---
with gr.Blocks(
    title="RecurrentGPT for AI Technical Books",
    css="footer {visibility: hidden}",
    theme="sudeepshouche/minimalist",
) as demo:
    gr.Markdown(
        """
    # RecurrentGPT for AI Technical Books
    Interactive Generation of Comprehensive AI Technical Books with Human-in-the-Loop
    """
    )

    # --- State Variables ---
    current_chapter = gr.State(1)
    book_instance = gr.State(None)
    recurrent_workflow = gr.State(None)
    human_workflow = gr.State(None)

    with gr.Tab("Auto-Generation"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            book_type = gr.Textbox(
                                label="Book Type",
                                placeholder="e.g. Machine Learning, Deep Learning",
                            )
                        with gr.Column(scale=2, min_width=400):
                            description = gr.Textbox(label="Description")
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            model_selection = gr.Radio(
                                ["gpt-4o", "gpt-4o-mini"],  # Corrected model names
                                label="Select Language Model",
                                value="gpt-4o-mini",
                            )
                        with gr.Column(scale=2, min_width=400):
                            language_selection = gr.Radio(
                                ["English", "Spanish", "Portuguese"],
                                label="Select Language",
                                value="English",
                            )
                    btn_init = gr.Button(
                        "Init Book Generation", variant="primary"
                    )
                    gr.Examples(
                        [
                            "Machine Learning",
                            "Deep Learning",
                            "Natural Language Processing",
                            "Computer Vision",
                            "Reinforcement Learning",
                            "AI Ethics",
                            "Neural Networks",
                            "Data Science",
                            "Robotics",
                        ],
                        inputs=[book_type],
                    )
                    written_paras = gr.Textbox(
                        label="Written Content (editable)",
                        max_lines=21,
                        lines=21,
                    )
                    with gr.Row():
                        gr.Markdown("### Readability Metrics\n")
                        clarity = gr.Textbox(label="Clarity")
                        flesch_score = gr.Number(
                            label="Flesch Reading Ease", precision=2
                        )
                        g_fog = gr.Number(
                            label="Gunning Fog Index", precision=2
                        )

                        calculate_button = gr.Button("Calculate Metrics")

            with gr.Column():
                with gr.Row():
                    gr.Markdown("### Memory Module\n")
                    short_memory = gr.Textbox(
                        label="Short-Term Memory (editable)",
                        max_lines=3,
                        lines=3,
                    )
                    long_memory = gr.Textbox(
                        label="Long-Term Memory (editable)",
                        max_lines=6,
                        lines=6,
                    )
                with gr.Row():
                    gr.Markdown("### Instruction Module\n")
                    with gr.Row():
                        instruction1 = gr.Textbox(
                            label="Instruction 1 (editable)",
                            max_lines=4,
                            lines=4,
                        )
                        instruction2 = gr.Textbox(
                            label="Instruction 2 (editable)",
                            max_lines=4,
                            lines=4,
                        )
                        instruction3 = gr.Textbox(
                            label="Instruction 3 (editable)",
                            max_lines=4,
                            lines=4,
                        )
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan = gr.Radio(
                                [
                                    "Instruction 1",
                                    "Instruction 2",
                                    "Instruction 3",
                                ],
                                label="Instruction Selection",
                            )
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction = gr.Textbox(
                                label="Selected Instruction (editable)",
                                max_lines=5,
                                lines=5,
                            )

                btn_step = gr.Button("Next Step", variant="primary")

        # Define Readability Metrics Calculation
        def update_metrics(text):
            (
                clarity_cat,
                flesch,
                g_fog_idx,
                ari,
                understand_cat,
                conciseness_cat,
            ) = calculate_readability_metrics(text)
            return clarity_cat, flesch, g_fog_idx

        # Initialize Book Generation
        async def initialize_book(
            book_type: str,
            description: str,
            language: str,
            model: str,
            # request: gr.Request,
        ):
            """Initializes the book generation process."""
            try:
                # Print the book type
                print(f"STEP 1")
                # Create a new book instance
                book = await Book.create(
                    title=book_type,
                    table_of_contents="To be generated...",
                    language=language,
                    model=model,
                )
                book_instance = book
                print(f"STEP 2")
                # Initialize the workflows
                recurrent_workflow = RecurrentGPTWorkflow(model_name=model, language=language)
                human_workflow = HumanWorkflow(model_name=model, language=language)
                print(f"STEP 3")
                # Prepare initial input data
                input_data = {
                    "book_type": book_type,
                    "description": description,
                    "input_topic": "Introduction",
                    "current_memory": "",
                    "related_topics": [],
                    "input_instruction": {
                        "Instruction_1": "Write an engaging introduction to the book.",
                        "Instruction_2": "Provide a brief overview of the topics covered.",
                        "Instruction_3": "Outline the target audience for the book.",
                    },
                    "new_concept_prompt": "",
                    "relevant_memories": "",
                }
                print(f"STEP 4")
                # Run the first step of the recurrent workflow
                recurrent_output = await recurrent_workflow.run_step(input_data)
                logger.               info(f"Recurrent Workflow Output: {recurrent_output}")
                print(f"STEP 5")
                # Update the Gradio interface with the initial output
                short_memory_text = recurrent_output["output_memory"]["updated_memory"]
                long_memory_text = ""
                written_paras_text = recurrent_output["output_summary"]
                instruction1_text = recurrent_output["output_instruction"]["Instruction_1"]
                instruction2_text = recurrent_output["output_instruction"]["Instruction_2"]
                instruction3_text = recurrent_output["output_instruction"]["Instruction_3"]
                print(f"STEP 6")
                return (
                    short_memory_text,
                    long_memory_text,
                    written_paras_text,
                    instruction1_text,
                    instruction2_text,
                    instruction3_text,
                    book_instance,
                    recurrent_workflow,
                    human_workflow,
                )
            except Exception as e:
                logger.error(f"Error initializing book: {e}")
                return "", "", "", "", "", "", None, None, None

        btn_init.click(
            initialize_book,
            inputs=[
                book_type,
                description,
                language_selection,
                model_selection,
                # gr.Request(),
            ],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                instruction1,
                instruction2,
                instruction3,
                book_instance,
                recurrent_workflow,
                human_workflow,
            ],
            queue=False,
        )

        # Execute Next Step
        async def execute_next_step(
            short_memory: str,
            long_memory: str,
            instruction1: str,
            instruction2: str,
            instruction3: str,
            written_paras: str,
            # request: gr.Request,
            response_file: str,
            book_instance: Book,
            recurrent_workflow: RecurrentGPTWorkflow,
            human_workflow: HumanWorkflow,
        ) -> Tuple[str, str, str, str, str, str, Book, RecurrentGPTWorkflow, HumanWorkflow]:
            """Executes the next step in the book generation process."""
            try:
                # Prepare input data for the next step
                input_data = {
                    "book_type": book_instance.title,
                    "description": description.value,
                    "input_topic": f"Chapter {current_chapter.value}",
                    "current_memory": short_memory,
                    "related_topics": recurrent_workflow.long_memory,
                    "input_instruction": {
                        "Instruction_1": instruction1,
                        "Instruction_2": instruction2,
                        "Instruction_3": instruction3,
                    },
                    "new_concept_prompt": "",
                    "relevant_memories": "",
                }

                # Run the recurrent workflow
                recurrent_output = await recurrent_workflow.run_step(input_data)
                logger.info(f"Recurrent Workflow Output: {recurrent_output}")

                # Generate and save the chapter
                await generate_chapter(
                    book_instance, current_chapter.value, input_data, recurrent_workflow
                )

                # Update the current chapter number
                current_chapter.value += 1

                # Update the Gradio interface with the output
                short_memory_text = recurrent_output["output_memory"]["updated_memory"]
                long_memory_text = "\n".join(recurrent_workflow.long_memory)
                written_paras_text = recurrent_output["output_summary"]
                instruction1_text = recurrent_output["output_instruction"]["Instruction_1"]
                instruction2_text = recurrent_output["output_instruction"]["Instruction_2"]
                instruction3_text = recurrent_output["output_instruction"]["Instruction_3"]

                return (
                    short_memory_text,
                    long_memory_text,
                    written_paras_text,
                    "Instruction 1",  # Reset selected plan
                    instruction1_text,
                    instruction2_text,
                    instruction3_text,
                    book_instance,
                    recurrent_workflow,
                    human_workflow,
                )
            except Exception as e:
                logger.error(f"Error executing next step: {e}")
                return "", "", "", "", "", "", None, None, None

        btn_step.click(
            execute_next_step,
            inputs=[
                short_memory,
                long_memory,
                instruction1,
                instruction2,
                instruction3,
                written_paras,
                # gr.Request(),
                gr.Textbox(label="Response File", visible=False),
                book_instance,
                recurrent_workflow,
                human_workflow,
            ],
            outputs=[
                short_memory,
                long_memory,
                written_paras,
                selected_plan,
                instruction1,
                instruction2,
                instruction3,
                book_instance,
                recurrent_workflow,
                human_workflow,
            ],
            queue=True,
        )

        # Calculate Readability Metrics
        calculate_button.click(
            update_metrics,
            inputs=[written_paras],
            outputs=[clarity, flesch_score, g_fog],
            queue=False,
        )

        # Link Radio Selection to Instruction Textbox
        selected_plan.select(
            on_select_plan,
            inputs=[selected_plan, instruction1, instruction2, instruction3],
            outputs=[selected_instruction],
            queue=False,
        )

    with gr.Tab("Human-in-the-Loop"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            book_type_human = gr.Textbox(
                                label="Book Type",
                                placeholder="e.g. Machine Learning, Deep Learning",
                                interactive=False,  # Disable input for this tab
                            )
                        with gr.Column(scale=2, min_width=400):
                            description_human = gr.Textbox(
                                label="Description", interactive=False
                            )
                    with gr.Row():
                        with gr.Column(scale=1, min_width=200):
                            model_selection_human = gr.Radio(
                                ["gpt-4o", "gpt-4o-mini"],  # Corrected model names
                                label="Select Language Model",
                                value="gpt-4o-mini",
                                interactive=False,
                            )
                        with gr.Column(scale=2, min_width=400):
                            language_selection_human = gr.Radio(
                                ["English", "Spanish", "Portuguese"],
                                label="Select Language",
                                value="English",
                                interactive=False,
                            )
                    btn_init_human = gr.Button(
                        "Init Book Generation", variant="primary", visible=False
                    )
                    written_paras_human = gr.Textbox(
                        label="Written Content (editable)",
                        max_lines=23,
                        lines=23,
                    )
            with gr.Column():
                with gr.Row():
                    gr.Markdown("### Memory Module\n")
                    short_memory_human = gr.Textbox(
                        label="Short-Term Memory (editable)",
                        max_lines=3,
                        lines=3,
                    )
                    long_memory_human = gr.Textbox(
                        label="Long-Term Memory (editable)",
                        max_lines=6,
                        lines=6,
                    )
                with gr.Row():
                    gr.Markdown("### Instruction Module\n")
                    with gr.Row():
                        instruction1_human = gr.Textbox(
                            label="Instruction 1",
                            max_lines=3,
                            lines=3,
                            interactive=False,
                        )
                        instruction2_human = gr.Textbox(
                            label="Instruction 2",
                            max_lines=3,
                            lines=3,
                            interactive=False,
                        )
                        instruction3_human = gr.Textbox(
                            label="Instruction 3",
                            max_lines=3,
                            lines=3,
                            interactive=False,
                        )
                    with gr.Row():
                        with gr.Column(scale=1, min_width=100):
                            selected_plan_human = gr.Radio(
                                [
                                    "Instruction 1",
                                    "Instruction 2",
                                    "Instruction 3",
                                ],
                                label="Instruction Selection",
                            )
                        with gr.Column(scale=3, min_width=300):
                            selected_instruction_human = gr.Textbox(
                                label="Selected Instruction (editable)",
                                max_lines=5,
                                lines=5,
                            )

                btn_step_human = gr.Button("Next Step", variant="primary")

        # Initialize Human-in-the-Loop Book Generation
        btn_init_human.click(
            lambda bt, d, l, m: initialize_book(bt, d, l, m),
            inputs=[
                book_type_human,
                description_human,
                language_selection_human,
                model_selection_human,
            ],
            outputs=[
                short_memory_human,
                long_memory_human,
                written_paras_human,
                instruction1_human,
                instruction2_human,
                instruction3_human,
                book_instance,
                recurrent_workflow,
                human_workflow,
            ],
            queue=False,
        )

        # Execute Next Step with Human Selection
        async def execute_human_step(
            short_memory: str,
            long_memory: str,
            selected_instruction: str,
            written_paras: str,
            # request: gr.Request,
            response_file: str,
            book_instance: Book,
            recurrent_workflow: RecurrentGPTWorkflow,
            human_workflow: HumanWorkflow,
        ) -> Tuple[str, str, str, str, str, str, Book, RecurrentGPTWorkflow, HumanWorkflow]:
            """Executes the next step in the book generation process with human control."""
            try:
                # Prepare input data for the next step
                input_data = {
                    "book_type": book_instance.title,
                    "description": description_human.value,
                    "input_topic": f"Chapter {current_chapter.value}",
                    "current_memory": short_memory,
                    "related_topics": recurrent_workflow.long_memory,
                    "input_instruction": {
                        "Instruction_1": selected_instruction,
                        "Instruction_2": "",
                        "Instruction_3": "",
                    },
                    "new_concept_prompt": "",
                    "relevant_memories": "",
                }

                # Run the recurrent workflow
                recurrent_output = await recurrent_workflow.run_step(input_data)
                logger.info(f"Recurrent Workflow Output: {recurrent_output}")

                # Generate and save the chapter
                await generate_chapter(
                    book_instance, current_chapter.value, input_data, recurrent_workflow
                )

                # Update the current chapter number
                current_chapter.value += 1

                # Update the Gradio interface with the output
                short_memory_text = recurrent_output["output_memory"]["updated_memory"]
                long_memory_text = "\n".join(recurrent_workflow.long_memory)
                written_paras_text = recurrent_output["output_summary"]
                instruction1_text = recurrent_output["output_instruction"]["Instruction_1"]
                instruction2_text = recurrent_output["output_instruction"]["Instruction_2"]
                instruction3_text = recurrent_output["output_instruction"]["Instruction_3"]

                return (
                    short_memory_text,
                    long_memory_text,
                    written_paras_text,
                    instruction1_text,
                    instruction2_text,
                    instruction3_text,
                    book_instance,
                    recurrent_workflow,
                    human_workflow,
                )
            except Exception as e:
                logger.error(f"Error executing human step: {e}")
                return "", "", "", "", "", "", None, None, None

        btn_step_human.click(
            execute_human_step,
            inputs=[
                short_memory_human,
                long_memory_human,
                selected_instruction_human,
                written_paras_human,
                # gr.Request(),
                gr.Textbox(label="Response File", visible=False),
                book_instance,
                recurrent_workflow,
                human_workflow,
            ],
            outputs=[
                short_memory_human,
                long_memory_human,
                written_paras_human,
                instruction1_human,
                instruction2_human,
                instruction3_human,
                book_instance,
                recurrent_workflow,
                human_workflow,
            ],
            queue=True,
        )
    # Print the concurrency count
    print(f"Concurrency Count: {config.CONCURRENCY_COUNT}")

    demo.queue(
        # concurrency_count=config.CONCURRENCY_COUNT
    )  # Increased concurrency for better performance


# --- Gradio Functions ---
def on_select_plan(
    selected_plan: str,
    instruction1: str,
    instruction2: str,
    instruction3: str,
) -> str:
    """Updates the selected instruction based on the radio button choice."""
    if selected_plan == "Instruction 1":
        return instruction1
    elif selected_plan == "Instruction 2":
        return instruction2
    elif selected_plan == "Instruction 3":
        return instruction3
    else:
        return ""


# --- Database Initialization ---
async def init_db():
    """Initializes the database by creating all tables."""
    await Tortoise.init(db_url=config.DATABASE_URL, modules={"models": ["__main__"]})
    await Tortoise.generate_schemas()
    logger.info("Database initialized and tables created.")


# --- Launch the App ---
if __name__ == "__main__":
    run_async(init_db())  # Initialize the database asynchronously
    demo.launch(
        max_threads=10,
    )