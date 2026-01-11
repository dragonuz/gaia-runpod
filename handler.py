import runpod
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
import base64
import io

# --- 1. CONFIGURACIÓN INICIAL (Cold Start) ---
# Esto se ejecuta una sola vez cuando el servidor "despierta".
print("🏗️ INICIANDO VULCAN: Cargando modelos SDXL...")

try:
    # Cargar el VAE (mejora los colores y detalles)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )

    # Cargar el Modelo Principal (SDXL 1.0 Base)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Mover a la GPU para máxima velocidad
    pipe.to("cuda")
    
    # Optimizaciones de memoria (opcional, pero recomendado)
    # pipe.enable_model_cpu_offload() 
    
    print("✅ VULCAN ONLINE: Modelos cargados y listos.")
    
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR MODELOS: {e}")
    raise e

# --- 2. EL MANEJADOR DE PETICIONES (Handler) ---
def handler(event):
    """
    Esta función se ejecuta cada vez que envías una orden desde tu App.
    """
    print("📩 Nueva misión recibida.")
    
    # Leer el input (si no hay prompt, usa uno por defecto)
    input_data = event.get("input", {})
    prompt = input_data.get("prompt", "A futuristic cyberpunk shark boat, neon lights, unreal engine 5 render, 8k")
    negative_prompt = input_data.get("negative_prompt", "low quality, blurry, distorted, ugly")
    
    try:
        # Generar la imagen
        print(f"🎨 Pintando: {prompt}")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30, # Calidad vs Velocidad (30 es buen balance)
            guidance_scale=7.5
        ).images[0]
        
        # Convertir imagen a Base64 para enviarla de vuelta
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print("🚀 Misión cumplida. Enviando imagen.")
        
        return {
            "status": "success",
            "image_base64": img_str
        }
        
    except Exception as e:
        print(f"❌ Error generando imagen: {e}")
        return {"status": "error", "message": str(e)}

# --- 3. INICIAR SERVIDOR ---
runpod.serverless.start({"handler": handler})
