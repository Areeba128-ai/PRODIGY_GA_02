🧠 Task 2: Text-to-Image Generation using Stable Diffusion
This project utilizes pre-trained generative models like Stable Diffusion to create images from natural language text prompts. By leveraging Hugging Face's 🤗 diffusers library and a GPU runtime in Google Colab, we can generate high-quality, contextually relevant images based on custom user input.

🚀 Objective
To generate context-aware images using Stable Diffusion v1.5 by simply providing a text prompt such as:

"a robot reading a book under a tree at sunset"

🛠️ Tools & Technologies
Tool/Library	Purpose
Python	Programming language
Google Colab	Cloud-based GPU environment
Hugging Face Diffusers	Text-to-image pipeline
Torch (PyTorch)	Deep learning backend
Matplotlib	Display images

✅ Setup Instructions
1. Open Google Colab
Visit https://colab.research.google.com/ and open a new notebook.

2. Enable GPU
Go to Runtime > Change runtime type

Set Hardware Accelerator to GPU

3. Install Required Libraries
bash
Copy
Edit
!pip install diffusers transformers accelerate scipy safetensors
4. Load the Stable Diffusion Model
python
Copy
Edit
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")
⚠️ If you don’t have GPU access, replace .to("cuda") with .to("cpu") and remove torch_dtype=torch.float16.

🖼️ Generate Images from Text
python
Copy
Edit
prompt = "a robot reading a book under a tree at sunset"
image = pipe(prompt).images[0]

# Display the image
plt.imshow(image)
plt.axis("off")
plt.show()
💾 Optional: Save the Image
python
Copy
Edit
image.save("robot_under_tree.png")
⚠️ Notes
Make sure you're logged into Hugging Face when asked.

You can get a free Hugging Face token from: https://huggingface.co/settings/tokens

Use short, clear prompts for best results.

Image generation may take ~10–30 seconds on GPU.

📌 Example Prompts
txt
Copy
Edit
"astronaut surfing on a wave in space"
"a medieval castle on a floating island"
"cyberpunk girl with neon lights"
📍 Output
You will get realistic or artistic images that align with your text prompt.

Images can be downloaded or displayed inline in Colab.

🤖 Model Used
Model: runwayml/stable-diffusion-v1-5

Base Framework: Hugging Face diffusers

