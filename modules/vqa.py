"""Visual Question Answering with ViLT (dandelin/vilt-b32-finetuned-vqa)."""
import io
import config
from utils.model_loader import safe_load


class VQAError(Exception):
    """Raised for invalid images or model failures."""


def load_image(uploaded_file):
    """Validate an uploaded file (JPG/JPEG/PNG, size limit, readable) and return an RGB PIL image."""
    from PIL import Image
    if uploaded_file is None:
        raise VQAError("Please upload an image first.")
    name = getattr(uploaded_file, "name", "image")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in config.ALLOWED_IMAGE_TYPES:
        raise VQAError(f"Unsupported file type '.{ext}'. Allowed: JPG, JPEG, PNG.")
    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if len(data) > config.MAX_IMAGE_MB * 1024 * 1024:
        raise VQAError(f"Image is larger than {config.MAX_IMAGE_MB} MB.")
    try:
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise VQAError("The file is not a valid image (corrupted or wrong format).") from exc
    return img


class VQAModel:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or config.VQA_MODEL_NAME
        self.device = device or config.get_device_name()

        def _load():
            import torch
            from transformers import ViltProcessor, ViltForQuestionAnswering
            processor = ViltProcessor.from_pretrained(self.model_name)
            model = ViltForQuestionAnswering.from_pretrained(self.model_name).to(self.device).eval()
            return torch, processor, model

        # Downloads on first use, then loads from the local cache. Retries on network errors.
        self._torch, self.processor, self.model = safe_load(_load, self.model_name, VQAError)

    def predict(self, image, question: str, top_k: int = 3) -> dict:
        """Return dict(answer, confidence, top) for an image + question."""
        if not question or not question.strip():
            raise VQAError("Please type a question about the image.")
        torch = self._torch
        try:
            enc = self.processor(image, question.strip(), return_tensors="pt", truncation=True,
                                 max_length=config.VQA_MAX_QUESTION_TOKENS)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits[0]
            probs = torch.sigmoid(logits)          # VQA is multi-label, so sigmoid
            vals, idxs = probs.topk(top_k)
        except Exception as exc:
            raise VQAError(f"VQA prediction failed: {exc}") from exc
        top = [(self.model.config.id2label[int(i)], float(v)) for v, i in zip(vals, idxs)]
        return {"answer": top[0][0], "confidence": top[0][1], "top": top}
