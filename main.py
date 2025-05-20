import os
import random
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import threading
from diffusers import StableDiffusionPipeline
import requests
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = 'https://openrouter.ai/api/v1/chat/completions'
TEXT_PROMT = "The image should depict an underwater scene where a fish is curiously observing a magical, glowing projection of a poppy flower. The poppy should appear vibrant and slightly surreal, with delicate red petals that emit a soft, bioluminescent glow, resembling the way light filters through water. The center of the flower should be a deep black, fading into a golden hue, mimicking the real-life details of a poppy but in a way that makes sense in an aquatic environment."

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

model_name = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")

# Дифузионка  1
# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe = pipe.to("cuda")

## GPT 2
# text_generator = pipeline("text-generation", model="gpt2")

# Load model directly
# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# model_name = "yahma/llama-13b-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )




mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh

segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
pose = mp_pose.Pose()
face_mesh = mp_face_mesh.FaceMesh()



static_backgrounds = {}
generated_backgrounds = deque(maxlen=5)
current_level = 0
sleep_timer = time.time()



fish_img = cv2.imread("fish.png", cv2.IMREAD_UNCHANGED)
poppies_img = cv2.imread("poppies.png", cv2.IMREAD_UNCHANGED)
static_bg_files = ["images/bg2.jpg", "images/bg3.jpg", "images/bg4.jpg", "images/bg6.jpg", "images/bg7.jpg", "images/bg8.jpg", "images/bg9.jpg", "images/bg10.jpg",
                   "images/bg11.jpg", "images/bg12.jpg"]
# def generate_explanation(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     output = model.generate(**inputs, max_length=100)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_prompts():
    data = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [
            {"role": "user", "content": "Write a 5 small PROMPTS for diff generate an image. "
                                        "ALL PROMTS MUST BE WRITTEN FROM A NEW LINE. You need to explain to "
                                        "the fish what a poppy looks like. The fish has never seen "
                                        "a poppy, but I want it to understand from the explanation and"
                                        " image of the poppy."}
        ]
    }
    try:
        response = requests.post(API_URL, json=data, headers=headers)
    # Проверка успешности запроса
        if response.status_code == 200:
            response_json = response.json()
            message_content = response_json['choices'][0]['message']['content']
            prompts = message_content.split('\n')
            return [prompt.strip() for prompt in prompts if prompt.strip()]
        else:
            return [TEXT_PROMT,TEXT_PROMT,TEXT_PROMT,TEXT_PROMT,TEXT_PROMT]
    except:
        print("Failed to fetch data from API.")
        return [TEXT_PROMT,TEXT_PROMT,TEXT_PROMT,TEXT_PROMT,TEXT_PROMT]


def generate_background(text_description):
    # text_description = generate_explanation(TEXT_PROMPT)[0]["generated_text"]
    # text_description = text_generator(TEXT_PROMPT, max_length=100, num_return_sequences=1)[0]["generated_text"]
    print(f"Generating background for {text_description}")
    try:
        image_result = pipe(
            text_description + ", using only red and turquoise colors",
            num_inference_steps=30
        )
        if not image_result.images:
            print("[ОШИБКА] Фон не сгенерировался")
            return None
        return np.array(image_result.images[0])[:, :, ::-1]
    except Exception as e:
        print(f"[ОШИБКА] Ошибка генерации фона: {e}")
        return None


def load_static_backgrounds():
    global static_backgrounds

    try:

        selected_files = random.sample(static_bg_files, 3)

        static_backgrounds = {i: cv2.imread(file) for i, file in enumerate(selected_files)}

        if any(bg is None for bg in static_backgrounds.values()):
            raise ValueError("Некоторые изображения не загрузились!")

        print("[ЗАГРУЗКА] 4 статических фона готовы!")

    except Exception as e:
        print(f"[ОШИБКА] Ошибка загрузки картинок фона: {e}")
        static_backgrounds = {i: cv2.imread(file) for i, file in enumerate(static_bg_files[:3])}


    time.sleep(60)


def background_generation():
    while True:
        prompts = generate_prompts()
        print(prompts)
        if prompts:
            for prompt in prompts:
                print(f"[ГЕНЕРАЦИЯ] Новый промт: {prompt}")
                new_bg = generate_background(prompt)
                if new_bg is not None:
                    generated_backgrounds.append(new_bg)
        time.sleep(200)

def pre_generate_backgrounds():
    print("[ГЕНЕРАЦИЯ] Предварительное создание 5 изображений...")
    prompts = generate_prompts()
    for pr in prompts:
        bg = generate_background(pr)
        if bg is not None:
            generated_backgrounds.append(bg)
    print("[ГЕНЕРАЦИЯ] Первые 5 фонов готовы!")
def update_backgrounds():
    while True:
        load_static_backgrounds()
        time.sleep(200)
load_static_backgrounds()
pre_generate_backgrounds()
threading.Thread(target=background_generation, daemon=True).start()
background_thread = threading.Thread(target=update_backgrounds, daemon=True)
background_thread.start()

cv2.namedWindow("Dream Simulation", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Dream Simulation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


cap = cv2.VideoCapture(0)
prev_x, prev_y = None, None
transition_frames = 5
fade_step = 0
current_background = None
next_background = None


while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("[ОШИБКА] Не удалось получить кадр с камеры")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if current_level < 3:
        new_background = static_backgrounds.get(current_level, np.zeros((h, w, 3), dtype=np.uint8))
    else:
        level = min(len(generated_backgrounds) - 1, current_level - 3)
        new_background = generated_backgrounds[level] if generated_backgrounds else np.zeros((h, w, 3), dtype=np.uint8)

    new_background_resized = cv2.resize(new_background, (w, h))


    if current_background is None:
        current_background = new_background_resized.copy()
        next_background = new_background_resized.copy()
    elif not np.array_equal(next_background, new_background_resized):
        next_background = new_background_resized.copy()
        fade_step = 0


    if fade_step < transition_frames:
        alpha = fade_step / transition_frames
        blended_background = cv2.addWeighted(current_background, 1 - alpha, next_background, alpha, 0)
        fade_step += 1
    else:
        blended_background = next_background.copy()
        current_background = next_background.copy()

    new_background_resized = blended_background


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_segment = segment.process(rgb_frame)
    results_pose = pose.process(rgb_frame)
    results_face = face_mesh.process(rgb_frame)


    mask = results_segment.segmentation_mask
    condition = (mask > 0.5).astype(np.uint8)


    if results_pose.pose_landmarks:
        nose = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_x, nose_y = int(nose.x * w), int(nose.y * h)

        if prev_x is not None and prev_y is not None:
            movement = abs(nose_x - prev_x) + abs(nose_y - prev_y)
            if movement < 5:
                if time.time() - sleep_timer > 10:
                    current_level += 1
                    print(f"[ПОГРУЖЕНИЕ] Уровень сна: {current_level}")
                    sleep_timer = time.time()
            else:
                current_level = max(0, current_level - 1)
                print(f"[ПРОБУЖДЕНИЕ] Сон становится обычным (уровень {current_level})")
                sleep_timer = time.time()

        prev_x, prev_y = nose_x, nose_y

    frame_no_bg = (frame * condition[:, :, None] + new_background_resized * (1 - condition[:, :, None])).astype(
        np.uint8)



    cv2.imshow("Dream Simulation", frame_no_bg)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
