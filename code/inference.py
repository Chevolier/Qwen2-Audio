import json
import base64
import torch
import librosa
import numpy as np
import os
import boto3
import traceback
import sys
from io import BytesIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.s3_client = None
        self.model_loaded = False

    def download_from_s3(self, s3_uri, local_path):
        """Download model files from S3"""
        try:
            if not self.s3_client:
                self.s3_client = boto3.client('s3')

            # Parse S3 URI
            if not s3_uri.startswith('s3://'):
                raise ValueError("S3 URI must start with 's3://'")

            s3_parts = s3_uri[5:].split('/', 1)
            bucket = s3_parts[0]
            prefix = s3_parts[1] if len(s3_parts) > 1 else ''

            logger.info(f"Downloading model from S3: {s3_uri}")

            # Create local directory
            os.makedirs(local_path, exist_ok=True)

            # List objects in the S3 prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            file_count = 0
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Skip directories
                        if key.endswith('/'):
                            continue

                        # Get relative path
                        rel_path = key[len(prefix):].lstrip('/')
                        local_file_path = os.path.join(local_path, rel_path)

                        # Create subdirectories if needed
                        local_dir = os.path.dirname(local_file_path)
                        if local_dir:
                            os.makedirs(local_dir, exist_ok=True)

                        # Download file
                        logger.info(f"Downloading {key} to {local_file_path}")
                        self.s3_client.download_file(bucket, key, local_file_path)
                        file_count += 1

            logger.info(f"Model download completed. Downloaded {file_count} files.")
            return True

        except Exception as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def load_model(self):
        """Load the Qwen2-Audio model and processor"""
        try:
            logger.info("Starting Qwen2-Audio model loading...")

            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")

            # Log library versions
            import transformers
            logger.info(f"Transformers version: {transformers.__version__}")
            logger.info(f"Torch version: {torch.__version__}")

            # Log GPU info if available
            if torch.cuda.is_available():
                logger.info(f"CUDA version: {torch.version.cuda}")
                logger.info(f"GPU count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                    logger.info(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

            # Get model configuration from environment variables
            model_id = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2-Audio-7B-Instruct")
            model_s3_uri = os.environ.get("MODEL_S3_URI", None)
            use_s3_model = os.environ.get("USE_S3_MODEL", "false").lower() == "true"

            logger.info(f"Model ID: {model_id}")
            logger.info(f"Use S3 Model: {use_s3_model}")
            if use_s3_model:
                logger.info(f"S3 Model URI: {model_s3_uri}")

            # Import transformers after logging
            logger.info("Importing transformers...")
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            logger.info("Transformers imported successfully")

            if use_s3_model and model_s3_uri:
                # Download model from S3
                local_model_path = "/tmp/model"
                logger.info(f"Downloading model from S3 to {local_model_path}")
                self.download_from_s3(model_s3_uri, local_model_path)
                model_path = local_model_path
                logger.info(f"Using S3 model from: {model_s3_uri}")
            else:
                # Use HuggingFace Hub
                model_path = model_id
                logger.info(f"Using HuggingFace model: {model_id}")

            # Load processor first
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=use_s3_model
            )
            logger.info("Processor loaded successfully")

            # Load model with more conservative settings
            logger.info("Loading model...")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=use_s3_model,
                low_cpu_mem_usage=True,
                max_memory={0: "20GiB"}  # Limit GPU memory usage
            )
            logger.info("Model loaded successfully")

            # Set model to eval mode
            self.model.eval()

            self.model_loaded = True
            logger.info("✅ Qwen2-Audio model loading completed successfully")

        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(traceback.format_exc())
            self.model_loaded = False
            raise

    def decode_audio(self, audio_data):
        """Decode base64 audio data"""
        try:
            logger.info("Decoding base64 audio data...")
            # Decode base64
            audio_bytes = base64.b64decode(audio_data)
            logger.info(f"Decoded audio bytes: {len(audio_bytes)}")

            # Load audio using librosa
            audio, sr = librosa.load(
                BytesIO(audio_bytes), 
                sr=self.processor.feature_extractor.sampling_rate
            )
            logger.info(f"Audio loaded: shape={audio.shape}, sr={sr}")

            return audio

        except Exception as e:
            logger.error(f"Error decoding audio: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def predict(self, data):
        """Generate prediction from input data"""
        try:
            if not self.model_loaded:
                raise RuntimeError("Model not loaded successfully")

            logger.info(f"Received prediction request: {type(data)}")
            logger.info(f"Request keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")

            # Parse input
            conversation = data.get("conversation", [])
            max_length = data.get("max_length", 256)
            temperature = data.get("temperature", 0.7)
            top_p = data.get("top_p", 0.9)

            logger.info(f"Conversation length: {len(conversation)}")
            logger.info(f"Generation params: max_length={max_length}, temperature={temperature}, top_p={top_p}")

            if not conversation:
                raise ValueError("No conversation provided")

            # Process conversation and extract audio
            audios = []
            for i, message in enumerate(conversation):
                logger.info(f"Processing message {i}: role={message.get('role', 'unknown')}")
                if isinstance(message.get("content"), list):
                    for j, content_item in enumerate(message["content"]):
                        logger.info(f"Processing content item {j}: type={content_item.get('type', 'unknown')}")
                        if content_item.get("type") == "audio":
                            # Handle base64 encoded audio
                            if "audio_base64" in content_item:
                                logger.info("Processing base64 audio")
                                audio = self.decode_audio(content_item["audio_base64"])
                                audios.append(audio)
                            # Handle audio URL (for testing)
                            elif "audio_url" in content_item:
                                logger.info(f"Processing audio URL: {content_item['audio_url']}")
                                from urllib.request import urlopen
                                audio = librosa.load(
                                    BytesIO(urlopen(content_item["audio_url"]).read()),
                                    sr=self.processor.feature_extractor.sampling_rate
                                )[0]
                                audios.append(audio)
                                logger.info(f"Audio from URL loaded: shape={audio.shape}")

            logger.info(f"Total audio inputs: {len(audios)}")

            # Apply chat template
            logger.info("Applying chat template...")
            text = self.processor.apply_chat_template(
                conversation, 
                add_generation_prompt=True, 
                tokenize=False
            )
            logger.info(f"Chat template applied. Text length: {len(text)}")
            logger.info(f"Text preview: {text[:200]}...")

            # Process inputs
            logger.info("Processing inputs...")
            inputs = self.processor(
                text=text, 
                audios=audios if audios else None, 
                return_tensors="pt", 
                padding=True
            )
            logger.info(f"Input IDs shape: {inputs.input_ids.shape}")

            # Move to device
            logger.info(f"Moving inputs to device: {self.device}")
            inputs.input_ids = inputs.input_ids.to(self.device)
            if hasattr(inputs, 'audio_features') and inputs.audio_features is not None:
                inputs.audio_features = inputs.audio_features.to(self.device)
                logger.info(f"Audio features shape: {inputs.audio_features.shape}")

            # Generate response with compatible parameters
            logger.info("Starting generation...")
            with torch.no_grad():
                # Use compatible generation parameters for transformers 4.37.0
                generation_config = {
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                    "pad_token_id": self.processor.tokenizer.eos_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                    "use_cache": True,
                    # Remove cache_position and other incompatible parameters
                }

                # Filter inputs to only pass compatible parameters
                model_inputs = {}
                if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
                    model_inputs['input_ids'] = inputs.input_ids
                if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
                    model_inputs['attention_mask'] = inputs.attention_mask.to(self.device)
                if hasattr(inputs, 'audio_features') and inputs.audio_features is not None:
                    model_inputs['audio_features'] = inputs.audio_features

                logger.info(f"Model input keys: {list(model_inputs.keys())}")

                generate_ids = self.model.generate(
                    **model_inputs,
                    **generation_config
                )

            logger.info(f"Generation completed. Output shape: {generate_ids.shape}")

            # Decode response
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

            result = {
                "generated_text": response,
                "input_length": inputs.input_ids.size(1),
                "output_length": generate_ids.size(1)
            }

            logger.info(f"✅ Prediction completed successfully")
            logger.info(f"Generated text: {response}")
            return result

        except Exception as e:
            logger.error(f"❌ Prediction error: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(traceback.format_exc())
            raise

# Global model handler
model_handler = ModelHandler()

def model_fn(model_dir):
    """Load model for SageMaker"""
    try:
        logger.info(f"Model function called with model_dir: {model_dir}")
        logger.info(f"Environment variables:")
        for key, value in os.environ.items():
            if key.startswith(('HF_', 'SAGEMAKER_', 'MODEL_', 'USE_')):
                logger.info(f"  {key}={value}")

        logger.info("Loading model...")
        model_handler.load_model()
        logger.info("Model function completed successfully")
        return model_handler

    except Exception as e:
        logger.error(f"❌ Model function error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def input_fn(request_body, request_content_type):
    """Parse input data"""
    try:
        logger.info(f"Input function called with content_type: {request_content_type}")
        logger.info(f"Request body type: {type(request_body)}")
        logger.info(f"Request body length: {len(request_body) if hasattr(request_body, '__len__') else 'Unknown'}")

        if request_content_type == "application/json":
            parsed_data = json.loads(request_body)
            logger.info(f"✅ Input parsed successfully")
            return parsed_data
        else:
            raise ValueError(f"Unsupported content type: {request_content_type}")

    except Exception as e:
        logger.error(f"❌ Input function error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def predict_fn(input_data, model):
    """Generate prediction"""
    try:
        logger.info("Predict function called")
        result = model.predict(input_data)
        logger.info("✅ Predict function completed successfully")
        return result

    except Exception as e:
        logger.error(f"❌ Predict function error: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def output_fn(prediction, accept):
    """Format output"""
    try:
        logger.info(f"Output function called with accept: {accept}")

        if accept == "application/json":
            result = json.dumps(prediction)
            logger.info("✅ Output function completed successfully")
            return result, accept
        else:
            raise ValueError(f"Unsupported accept type: {accept}")

    except Exception as e:
        logger.error(f"❌ Output function error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
