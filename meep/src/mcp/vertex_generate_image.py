
from typing import Iterable, List, Optional, Literal

import os, asyncio, vertexai.vision_models
import mcp.vision

BUCKET_NAME = "vertex_test_storage"

def gcs_to_http_url(gcs_url):
    # Suppression du préfixe 'gs://'
    if not gcs_url.startswith("gs://"):
        raise ValueError("L'URL doit commencer par 'gs://'")
    path = gcs_url[5:]
    parts = path.split("/", 1)
    if len(parts) != 2:
        raise ValueError("L'URL GCS n'est pas valide")
    bucket, file_path = parts
    return f"https://storage.googleapis.com/{bucket}/{file_path}"

service_account_path = os.path.join(os.getcwd(), "data", "credentials", "service_account.json")
if not os.path.exists(service_account_path):  # XXX : Service account disabled for now
    pass # raise FileNotFoundError(f"Le fichier de compte de service n'a pas été trouvé : {service_account_path}")
else:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path

GENERATION_MODELS = {}  # XXX : Service account disabled for now
#     "fast": vertexai.vision_models.ImageGenerationModel.from_pretrained(f"imagen-4.0-fast-generate-001"),
#     "normal": vertexai.vision_models.ImageGenerationModel.from_pretrained(f"imagen-4.0-generate-001"),
#     "ultra": vertexai.vision_models.ImageGenerationModel.from_pretrained(f"imagen-4.0-ultra-generate-001"),
# }
gcs_uri = f"gs://{BUCKET_NAME}"

async def generate_image_using_vertex(
    prompt: str,
    mode: Literal["fast", "normal", "ultra"] = "fast",
    negative_prompt: Optional[str] = None,
    number_of_images: int = 1,
) -> List[str]:

    generation_model = GENERATION_MODELS[mode]

    response: Iterable[vertexai.vision_models.GeneratedImage] = await asyncio.to_thread(
        generation_model.generate_images,
        prompt=prompt,
        number_of_images=number_of_images,
        #aspect_ratio="1:1",
        negative_prompt=negative_prompt,
        output_gcs_uri=gcs_uri,
        add_watermark=False,
        safety_filter_level="block_only_high",
        person_generation="allow_all"
    )
    return [gcs_to_http_url(image._gcs_uri) for image in response]

async def generate_image(prompt: str, requestor: str, negative_prompt: Optional[str] = None, mode: Literal["fast", "normal", "ultra"] = "fast"):
    """Generate an image from a prompt using the Google Generative AI API. As requestor write down the name of the one who requested the image exactly as it appears in the chat."""
    urls: List[str] = await generate_image_using_vertex(prompt, mode=mode, negative_prompt=negative_prompt, number_of_images=1)
    try:
        results: List[str] = await asyncio.gather(*[mcp.vision.process_image(url) for url in urls], return_exceptions=True)
        results = str({url: result for url, result in zip(urls, results)})  # Convert the list of results to a string for display
    except Exception as e:
        results = f"Error processing image: {str(e)}. Generated the following:" + str(urls)

    return "Images generated successfully with prompt: " + prompt + f" (requested by {requestor}). Generations: {results}."

tools: List[callable] = [
    generate_image,
]
