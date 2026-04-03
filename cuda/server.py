from dotenv import load_dotenv
import os
import sys
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import HTMLResponse
import uvicorn
import numpy as np
import cv2
import asyncio
# from paddleocr import PaddleOCR
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from io import BytesIO
from pydantic import BaseModel
from rapidocr import RapidOCR  # Paddle的cuda镜像太大，改用torch，RapidOCR支持torch
from transformers import AutoModel, AutoProcessor
ImageFile.LOAD_TRUNCATED_IMAGES = True

on_linux = sys.platform.startswith('linux')

load_dotenv()
app = FastAPI()

api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")
http_port = int(os.getenv("HTTP_PORT", "8060"))
server_restart_time = int(os.getenv("SERVER_RESTART_TIME", "300"))
env_auto_load_txt_modal = os.getenv("AUTO_LOAD_TXT_MODAL", "off") == "on" # 是否自动加载CLIP文本模型，开启可以优化第一次搜索时的响应速度,文本模型占用700多m内存

clip_model_name = os.getenv("CLIP_MODEL")
legacy_clip_model_aliases = {
    # cn-clip 历史模型名 → transformers 版 Chinese-CLIP（保持中文能力）
    "ViT-B-16": "OFA-Sys/chinese-clip-vit-base-patch16",
    "ViT-L-14": "OFA-Sys/chinese-clip-vit-large-patch14",
    "ViT-H-14": "OFA-Sys/chinese-clip-vit-huge-patch14",
}


ocr_model = None
clip_processor = None
clip_model = None
resolved_clip_model_name = None

restart_task = None
restart_lock = asyncio.Lock()
_clip_load_lock = __import__('threading').Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"

class ClipTxtRequest(BaseModel):
    text: str

def load_ocr_model():
    global ocr_model
    if ocr_model is None:
        ocr_model = RapidOCR(
            params={
                "Global.with_torch": True,
                "EngineConfig.torch.use_cuda": torch.cuda.is_available(),  # 使用torch GPU版推理
                # "EngineConfig.torch.gpu_id": 0,  # 指定GPU id
            }
        )
        # https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/

def load_clip_model():
    global clip_processor
    global clip_model
    global resolved_clip_model_name
    if not clip_model_name:
        raise RuntimeError("CLIP_MODEL is required, for example: google/siglip2-base-patch16-224")
    if clip_processor is not None:
        return
    selected_model_name = legacy_clip_model_aliases.get(clip_model_name, clip_model_name)
    with _clip_load_lock:
        if clip_processor is not None:
            return  # double-check after acquiring lock
        if selected_model_name != clip_model_name:
            print(f"[compat] CLIP_MODEL={clip_model_name} is a legacy cn-clip alias. Using {selected_model_name} instead.")
        try:
            model = AutoModel.from_pretrained(selected_model_name)
        except Exception as e:
            hf_endpoint = os.getenv("HF_ENDPOINT", "")
            hint = (
                f"Failed to load model '{selected_model_name}': {e}\n"
                "Hint: If you are behind a firewall, set HF_ENDPOINT (e.g. https://hf-mirror.com) "
                "or download the model manually and set CLIP_MODEL to the local path."
            )
            if hf_endpoint:
                hint += f"\nCurrent HF_ENDPOINT={hf_endpoint}"
            raise RuntimeError(hint) from e
        model.eval()
        clip_model = model.to(device)
        clip_processor = AutoProcessor.from_pretrained(selected_model_name)
        resolved_clip_model_name = selected_model_name


def get_normalized_image_features(image_obj):
    inputs = clip_processor(images=image_obj, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        if hasattr(clip_model, "get_image_features"):
            image_features = clip_model.get_image_features(**inputs)
        else:
            outputs = clip_model(**inputs)
            image_features = getattr(outputs, "image_embeds", None)
            if image_features is None:
                raise RuntimeError("current model does not provide image embeddings")
    return F.normalize(image_features, dim=-1)


def get_normalized_text_features(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        if hasattr(clip_model, "get_text_features"):
            text_features = clip_model.get_text_features(**inputs)
        else:
            outputs = clip_model(**inputs)
            text_features = getattr(outputs, "text_embeds", None)
            if text_features is None:
                raise RuntimeError("current model does not provide text embeddings")
    return F.normalize(text_features, dim=-1)

@app.on_event("startup")
async def startup_event():
    if env_auto_load_txt_modal:
        load_clip_model()

@app.on_event("shutdown")
async def on_shutdown():
    if restart_task and not restart_task.done():
        restart_task.cancel()
        try:
            await restart_task
        except asyncio.CancelledError:
            pass

async def restart_timer():
    await asyncio.sleep(server_restart_time)
    restart_program()

@app.middleware("http")
async def activity_monitor(request, call_next):
    global restart_task

    async with restart_lock:
        if restart_task and not restart_task.done():
            restart_task.cancel()

        restart_task = asyncio.create_task(restart_timer())

    response = await call_next(request)
    return response


async def verify_header(api_key: str = Header(...)):
    # 在这里编写验证逻辑，例如检查 api_key 是否有效
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


def to_fixed(num):
    return str(round(num, 2))

def convert_rapidocr_to_json(rapidocr_output):

    if rapidocr_output.txts is None:
        return {'texts': [], 'scores': [], 'boxes': []}

    texts = list(rapidocr_output.txts)
    scores = [f"{score:.2f}" for score in rapidocr_output.scores]
    boxes_coords = rapidocr_output.boxes

    boxes = []
    for box in boxes_coords:
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        width = x_max - x_min
        height = y_max - y_min

        boxes.append({
            'x': to_fixed(x_min),
            'y': to_fixed(y_min),
            'width': to_fixed(width),
            'height': to_fixed(height)
        })

    output = {
        'texts': texts,
        'scores': scores,
        'boxes': boxes
    }

    return output

@app.get("/", response_class=HTMLResponse)
async def top_info():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)):
    return {
        'result': 'pass',
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "device": device,
        "clip_model": clip_model_name,
        "clip_model_resolved": resolved_clip_model_name or legacy_clip_model_aliases.get(clip_model_name, clip_model_name)
    }


@app.post("/restart")
async def check_req(api_key: str = Depends(verify_header)):
    # cuda版本 OCR没有显存未释放问题，这边可以关闭重启
    return {'result': 'unsupported'}
    # restart_program()

@app.post("/restart_v2")
async def check_req(api_key: str = Depends(verify_header)):
    # 预留触发服务重启接口-自动释放内存
    restart_program()
    return {'result': 'pass'}

@app.post("/ocr")
async def process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    load_ocr_model()
    image_bytes = await file.read()
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        _result = await predict(ocr_model, img)
        result = convert_rapidocr_to_json(_result)
        del img
        del _result
        return {'result': result}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}

@app.post("/clip/img")
async def clip_process_image(file: UploadFile = File(...), api_key: str = Depends(verify_header)):
    load_clip_model()
    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_features = get_normalized_image_features(image)
        return {'result': ["{:.16f}".format(vec) for vec in image_features[0]]}
    except Exception as e:
        print(e)
        return {'result': [], 'msg': str(e)}

@app.post("/clip/txt")
async def clip_process_txt(request:ClipTxtRequest, api_key: str = Depends(verify_header)):
    load_clip_model()
    text_features = get_normalized_text_features(request.text)
    return {'result': ["{:.16f}".format(vec) for vec in text_features[0]]}

async def predict(predict_func, inputs):
    return await asyncio.get_running_loop().run_in_executor(None, predict_func, inputs)


def restart_program():
    print("restart_program")
    python = sys.executable
    os.execl(python, python, *sys.argv)


if __name__ == "__main__":
    uvicorn.run("server:app", host=None, port=http_port)
