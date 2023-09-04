import cv2
import os
from ultralytics import YOLO
from pycocotools.coco import COCO
from transformers import AutoTokenizer, AutoConfig, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from huggingface_hub import login

# Load COCO class info
coco = COCO('coco_classes/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())

# Create dictionary to map class ID to name
coco_classes = {}
for cat in cats:
    id = cat['id'] - 1
    name = cat['name']
    coco_classes[id] = name

yolo_model = YOLO("weights/yolov8s.pt")

input_path = 'images'
conf = 0.6

name = "meta-llama/Llama-2-13b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
)

# Check if the input path points to a directory or a video file
descriptions = []
if os.path.isdir(input_path):
    for image_file in os.listdir(input_path):
        image_path = os.path.join(input_path, image_file)
        image = cv2.imread(image_path)

        # Handle case when image is not loaded correctly
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue

        image = cv2.resize(image, (640, 640))
        results = yolo_model(image)[0]

        prompt_elements = []
        for result in results:
            box = result.boxes.xyxy
            class_id = int(result.boxes.cls)
            class_name = coco_classes.get(class_id, "Unknown")

            if result.boxes.conf >= conf:
                xmin, ymin, xmax, ymax = [int(coord) for coord in box[0]]

                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2

                img_height, img_width = image.shape[:2]

                x_pos = "left" if x_center < img_width / 2 else "right"
                y_pos = "top" if y_center < img_height / 2 else "bottom"
                
                position = f"{y_pos} {x_pos}"
                prompt_elements.append(f"a {class_name} {position}")

        # Convert list of descriptions into a single string
        prompt_temp = "Given the setting of  " + ", ".join(prompt_elements) + "."
        music_prompt = f"{prompt_temp}. suggest a suitable playlist or songs."
        # Get playlist suggestions from LLM
        music_recommendation = generation_pipe(
            music_prompt,
            max_length=300,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=10,
            temperature=0.4,
            top_p=0.9
        )
        
        # Extract the recommended songs or genres from the response
        recommended_playlist = music_recommendation[0]['generated_text']

        print(recommended_playlist)
        