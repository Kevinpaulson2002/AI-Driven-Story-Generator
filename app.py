import os
from flask import Flask, render_template, request
import google.generativeai as genai
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

import base64
from io import BytesIO
#Make you won .env file and store your API KEYS there
load_dotenv()

app = Flask(__name__)


api_key = os.getenv("GOOGLE_API_KEY")  
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY") 

if not api_key or not huggingface_api_key:
    raise ValueError("API keys are missing. Please set them in the .env file.")

genai.configure(api_key=api_key)


generation_config = {
    "temperature": 2,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)


def generate_story(keywords, description):
    prompt = f"You are a story generator tool for kids ,Generate a story based on the following keywords: {keywords}. Story theme: {description}. The story should be about 400 to 500 words long."

    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    story = response.text if response else "No story generated. Please try again."

    prompt = f"Can you give a good scene (ONLY ONE SCENE IS NEEDED THAT BEST DESCRIBES THE STORY) as a line of text for prompting a text-to-image generator AI?"
    core_scene_response = chat_session.send_message(prompt)
    core_scene = core_scene_response.text if core_scene_response else "No scene generated."

    prompt = f"Give a good title for the story(JUST ONE TITLE IS ENOUGH THAT bEST DESCRIBE THE STORY)."
    title_response = chat_session.send_message(prompt)
    story_title = title_response.text if title_response else "Untitled Story"

    return story, core_scene, story_title


def generate_image_from_story(core_story_line_for_image_generation_prompt):
    client = InferenceClient(model="stabilityai/stable-diffusion-3-medium-diffusers", token=huggingface_api_key)

    
    image = client.text_to_image(core_story_line_for_image_generation_prompt)

    
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{image_base64}"

@app.route('/', methods=['GET', 'POST'])
def index():
    story = ""
    image_url = None
    story_title=None
    if request.method == 'POST':
        keywords = request.form['keywords']
        description = request.form['description']
        
        # Generate the story,cut-or-core-scene-for image generation,story tile
        story, core_story_line_for_image_generation_prompt,story_title = generate_story(keywords, description)

        # Generate image based on the core storyline
        image_url = generate_image_from_story(core_story_line_for_image_generation_prompt)

    return render_template('index.html', story=story,story_title=story_title, image_url=image_url)

if __name__ == '__main__':
    app.run(debug=True)
