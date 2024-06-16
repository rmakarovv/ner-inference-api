import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Constants
MODEL_PATH = "ner_bert.bin"
LABEL_LIST = ['O', 'B-discount', 'B-value', 'I-value']

# Initialize FastAPI app
app = FastAPI()

description = """
## Описание

Данный сервис предоставляет возможность инференеса NER модели для русского языка со следующими тэгами: {'O', 'B-discount', 'B-value', 'I-value'}

## Использование

Для получения результата используйте GET запрос на '/inference' с json файлом, содержащим текст для инференса: {'text': '...'}

Результат - словарь с двумя полями: 'tokens' (токенизированный текст) и 'labels' (присвоенные NER значения)

Связаться:
[@RomanMakarov](https://t.me/RomanMakar0v)
"""

app = FastAPI(
    title="NER таггинг",
    description=description,
    docs_url='/',
)

# Load model and tokenizer
@app.on_event("startup")
def load_model():
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()

class InferenceRequest(BaseModel):
    text: str

class InferenceResponse(BaseModel):
    tokens: list
    labels: list

def inference_single(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt')
    tokens = {k: v.to(model.device) for k, v in tokens.items()}

    with torch.no_grad():
        pred = model(**tokens)

    indices = pred.logits.argmax(dim=-1)[0].cpu().numpy()
    token_text = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

    toks, ids = [], []
    for i in range(len(token_text)):
        if i and token_text[i][:2] == '##':
            toks[-1] += token_text[i][2:]
        elif i and LABEL_LIST[indices[i]] == 'I-value' and LABEL_LIST[indices[i - 1]] != 'B-value':
            toks.append(token_text[i])
            ids.append(0)
        else:
            toks.append(token_text[i])
            ids.append(indices[i])

    toks = toks[1:-1]
    ids = ids[1:-1]

    return toks, ids

@app.get("/inference", response_model=InferenceResponse)
def get_inference(request: InferenceRequest):
    text = request.text
    if not text:
        raise HTTPException(status_code=400, detail="Input text is required")

    toks, ids = inference_single(text, tokenizer, model)
    labels = [LABEL_LIST[idx] for idx in ids]
    return InferenceResponse(tokens=toks, labels=labels)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
