import runpod
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, EulerAncestralDiscreteScheduler
import base64
import io

print("🎨 INICIANDO GAIA (SDXL Engine)...")

# --- CARGA DEL MODELO (Optimizado en FP16) ---
try:
    # 1. Cargar VAE (Mejora colores y bordes, vital para 3D)
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16
    )

    # 2. Cargar Modelo Base
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # 3. Scheduler Rápido (Euler A es rápido y creativo)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Mover a GPU
    pipe.to("cuda")
    
    # Compilar para velocidad extra (Opcional, puede tardar el primer inicio)
    # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    print("✅ GAIA ONLINE: Listo para pintar.")

except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")
    raise e

def handler(event):
    input_data = event.get("input", {})
    
    # --- ENTRADA DEL USUARIO ---
    user_prompt = input_data.get("prompt", "a futuristic tank")
    
    # --- ⚙️ OPTIMIZACIÓN 3D (SECRET SAUCE) ---
    # Inyectamos esto para asegurar que VULCAN entienda la imagen después
    forced_suffix = ", white background, 3d render style, orthographic view, 4k, high quality, studio lighting, clean geometry"
    full_prompt = user_prompt + forced_suffix
    
    # Negative Prompt para evitar ruido y fondos complejos
    negative_prompt = "shadows, complex background, noise, messy, blurry, low quality, text, watermark, human, organic"

    try:
        print(f"🖌️ Generando Blueprint: {user_prompt}")
        
        # Generación Estricta 1024x1024 (Nativo de SDXL y perfecto para TripoSR)
        image = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            height=1024,
            width=1024,
            num_inference_steps=30, # 30 pasos es el equilibrio calidad/velocidad
            guidance_scale=7.0
        ).images[0]
        
        # Convertir a Base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print("🚀 Blueprint Terminado.")
        
        return {
            "status": "success",
            "image_base64": img_str
        }

    except Exception as e:
        print(f"❌ Error en GAIA: {e}")
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
