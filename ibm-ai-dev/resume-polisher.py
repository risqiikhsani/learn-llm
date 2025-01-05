from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
)
from langchain.prompts import PromptTemplate
import gradio as gr

model = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

llm = ChatHuggingFace(llm=model)

# Function to polish the resume using the model, making polish_prompt optional
def polish_resume(position_name, resume_content, polish_prompt=""):
    # Check if polish_prompt is provided and adjust the combined_prompt accordingly
    if polish_prompt and polish_prompt.strip():
        prompt_use = f"Given the resume content: '{resume_content}', polish it based on the following instructions: {polish_prompt} for the {position_name} position."
    else:
        prompt_use = f"Suggest improvements for the following resume content: '{resume_content}' to better align with the requirements and expectations of a {position_name} position. Return the polished version, highlighting necessary adjustments for clarity, relevance, and impact in relation to the targeted role."

    # Generate a response using the model with parameters
    generated_response = llm.invoke(prompt_use, max_tokens=512, temperature=0.7)

    # Extract and return the generated text
    generated_text = generated_response.content
    return generated_text

# Create Gradio interface for the resume polish application, marking polish_prompt as optional
resume_polish_application = gr.Interface(
    fn=polish_resume,
    allow_flagging="never", # Deactivate the flag function in gradio as it is not needed.
    inputs=[
        gr.Textbox(label="Position Name", placeholder="Enter the name of the position..."),
        gr.Textbox(label="Resume Content", placeholder="Paste your resume content here...", lines=20),
        gr.Textbox(label="Polish Instruction (Optional)", placeholder="Enter specific instructions or areas for improvement (optional)...", lines=2),
    ],
    outputs=gr.Textbox(label="Polished Content"),
    title="Resume Polish Application",
    description="This application helps you polish your resume. Enter the position your want to apply, your resume content, and specific instructions or areas for improvement (optional), then get a polished version of your content."
)

# Launch the application
resume_polish_application.launch()
