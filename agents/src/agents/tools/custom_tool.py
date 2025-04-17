from crewai.tools import BaseTool
from google import genai
from google.genai import types
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

class GeminiImageDirectTool(BaseTool):
    name: str = "generate_image_direct"
    description: str = "Generate images using Gemini Pro model based on the prompt"

    def _run(self, prompt: str, num_images: int = 1) -> str:
        try:
            # enhanced_prompt = f"""
            # Create a detailed image based on this description:
            # {prompt}
            
            # Please provide a detailed description of the image that would be generated.
            # Focus on visual elements, composition, colors, and style.
            # """
            
            response = client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images= num_images,
                )
            )
            for generated_image in response.generated_images:
                image = Image.open(BytesIO(generated_image.image.image_bytes))
                image.save('gemini-native-image.png')
                image.show()
                
            return 'gemini-native-image.png'
        except Exception as e:
            return f"Erro ao gerar a imagem: {e}"