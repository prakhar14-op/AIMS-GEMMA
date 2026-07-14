
import torch
import logging
from transformers import AutoProcessor, AutoModelForPreTraining
logger = logging.getLogger(__name__)

class GemmaAnalyzer:
    def __init__(self, model_id="google/gemma-2-2b-it"): # Ensure this matches your downloaded model
        logger.info(f"Loading model: {model_id}...")
        # Use AutoModelForPreTraining for most modern multimodal Gemma models
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForPreTraining.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
    def analyze_event(self, frames, detections):
        # 1. Prompt Assembly: Formatting the 5-frame sequence [cite: 130]
        prompt = f"""
        <|system|>
        You are a Forensic Security AI. Analyze these 5 chronological frames. 
        Detect forced entry or threats. Output ONLY JSON. 
        Schema: {{"threat_level": int, "reasoning": str, "trigger_alarm": bool}}
        <|user|>
        Analyze this sequence: {['<image>'] * len(frames)}
        """
        
        # 2. Processor Application: Create tokenized input [cite: 141]
        inputs = self.processor(text=prompt, images=frames, return_tensors="pt").to("cuda")
        
        # 3. Cognitive Burn (Forward Pass) 
        output_ids = self.model.generate(**inputs, max_new_tokens=200)
        result = self.processor.batch_decode(output_ids, skip_special_tokens=True)
        
        return result[0]