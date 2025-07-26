import logging
from typing import List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.cerebras import Cerebras
from .functions.metadata_to_pdf import create_dpp_pdf

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic Models ---
# Define models at the module level for better structure and reusability.
class Question(BaseModel):
    """Represents a single question in the DPP."""
    text: str

class DPP(BaseModel):
    """Defines the structure for the AI-generated DPP content."""
    topic: str
    language: str
    instructions: str
    questions: List[Question]

# --- DPPify Agent ---
class DPPify:
    """
    An agent class to generate Daily Practice Problem (DPP) PDFs.
    It orchestrates AI content generation and PDF creation.
    """

    # Use a dictionary for clean and scalable prompt selection
    PROMPT_MAP = {
        "onlymcq": "backend/prompts/only_mcq_creator_system_prompt.txt",
        "onlysaq": "backend/prompts/only_saq_creator_system_prompt.txt",
        "both": "backend/prompts/both_questions_creater_system_prompt.txt",
    }

    def _get_system_prompt(self, question_type: str) -> str:
        """
        Selects and reads the appropriate system prompt file based on question type.
        
        Args:
            question_type (str): The type of questions required (e.g., "Only MCQ").
            
        Returns:
            str: The content of the system prompt.
            
        Raises:
            FileNotFoundError: If the prompt file does not exist.
        """
        normalized_q_type = question_type.lower().replace(" ", "")
        # Default to 'both' if the type is unknown
        prompt_path = self.PROMPT_MAP.get(normalized_q_type, self.PROMPT_MAP["both"])
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            logging.error(f"Prompt file not found at path: {prompt_path}")
            raise FileNotFoundError(f"Could not find the required prompt file: {prompt_path}")

    def _generate_dpp_metadata(self, topic_name: str, total_questions: int, question_type: str, difficulty_level: str, api_key: str, language: str,additional_instruction: str) -> dict:
        """
        Generates DPP content using an AI model.
        
        Returns:
            dict: A dictionary containing the topic, language, instructions, and questions.
            
        Raises:
            Exception: For API errors or if the model returns invalid data.
        """
        system_prompt = self._get_system_prompt(question_type)

        # Dynamically create a response model to include the number of questions in the Pydantic Field description.
        # This helps guide the LLM to generate the correct number of items.
        class DynamicDPP(DPP):
            questions: List[Question] = Field(description=f"A list of exactly {total_questions} questions.")

        user_prompt = f"""Generate a Daily Practice Problem (DPP) sheet with ABSOLUTE PRECISION using ONLY the following specifications:

TOPIC: {topic_name}
LANGUAGE: {language}
EXACT QUESTION COUNT: {total_questions} (non-negotiable - must be exactly this number)
DIFFICULTY: {difficulty_level}
ADDITIONAL INSTRUCTIONS: {additional_instruction}

CRITICAL EXECUTION RULES:
1. FOLLOW ADDITIONAL INSTRUCTIONS TO THE LETTER - These override all other considerations
2. Generate EXACTLY {total_questions} questions - no more, no less (validate count before output)
3. All content MUST be in {language} without any exceptions
4. Difficulty level {difficulty_level} must be consistently maintained throughout
5. Structure MUST include:
   - Topic header with {topic_name}
   - Clear instructions section incorporating ALL additional instructions
   - Numbered questions (1 to {total_questions}) with no extra content

MANDATORY COMPLIANCE CHECKS:
✓ Verify every additional instruction is implemented
✓ Confirm question count is precisely {total_questions}
✓ Ensure zero content outside specifications
✓ Validate all text is in {language}
✓ Cross-check difficulty consistency

OUTPUT REQUIREMENTS:
- JSON format matching DPP schema EXCLUSIVELY
- 'questions' array must contain exactly {total_questions} items
- NO explanations, disclaimers, or extra text
- STRICTLY follow additional instructions above all else

WARNING: Any deviation from specifications or additional instructions will invalidate the output. Prioritize instruction compliance over creativity."""
        
        agent = Agent(
            model=Cerebras(id="qwen-3-235b-a22b", api_key=api_key,max_completion_tokens=40000),
            system_message=system_prompt,
            response_model=DynamicDPP,
        )

        try:
            logging.info(f"Generating DPP for topic: {topic_name}")
            response = agent.run(user_prompt).content
        except Exception as e:
            logging.error(f"AI model API call failed: {e}")
            raise ConnectionError(f"Failed to get a response from the AI model. Please check your API key and network connection. Details: {e}")

        if not response or not response.questions:
            logging.error("AI model returned an empty or invalid response.")
            raise ValueError("The AI model failed to generate questions. Please try again with a different topic or settings.")

        # Process questions locally, avoiding global state
        questions_list = [q.text for q in response.questions]

        return {
            "topic_name": response.topic,
            "dpp_language": response.language,
            "instruction": response.instructions,
            "questions": questions_list
        }

    def run(self, topic_name: str, question_type: str, total_q: int, level: str, api_key: str,dpp_language: str = "English",additional_instruction: str = "") -> str:
        """
        The main execution method to generate and save a DPP PDF.
        
        Args:
            topic_name (str): The topic for the DPP.
            question_type (str): The type of questions.
            total_q (int): The total number of questions.
            level (str): The difficulty level.
            api_key (str): The Cerebras API key.
            
        Returns:
            str: The file path to the generated PDF.
        """
        try:
            # Step 1: Generate content from the AI
            dpp_metadata = self._generate_dpp_metadata(
                topic_name=topic_name,
                question_type=question_type,
                total_questions=total_q,
                difficulty_level=level,
                api_key=api_key,
                language=dpp_language,
                additional_instruction=additional_instruction
            )

            # Step 2: Create the PDF from the generated content
            logging.info("Creating PDF from generated metadata.")
            pdf_path = create_dpp_pdf(
                topic_name=dpp_metadata["topic_name"],
                questions=dpp_metadata["questions"],
                instructions=dpp_metadata["instruction"],
                total_q=total_q,
            )

            logging.info(f"Successfully created DPP PDF at: {pdf_path}")
            return pdf_path
            
        except (FileNotFoundError, ConnectionError, ValueError) as e:
            # Re-raise known errors with clear messages for the UI
            raise e
        except Exception as e:
            # Catch any other unexpected errors
            logging.error(f"An unexpected error occurred in the DPP generation process: {e}")
            raise RuntimeError(f"An unexpected error occurred: {e}")
