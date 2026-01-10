import runpod
from diffusers import StableDiffusionXLPipeline
import torch
from PIL import Image
import base64
from io import BytesIO

# ==========================================
# 1. CARGAR EL MODELO (Solo al inicio)
# ==========================================
print("🌍 Iniciando GAIA: Cargando modelo SDXL Turbo...")

# Usamos SDXL Turbo porque es rapidísimo (ideal para prototipado)
# y genera texturas nítidas perfectas para heightmaps.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
)

# Mover el modelo a la GPU
pipe.to("cuda")

print("✅ GAIA: Modelo cargado y listo en GPU.")

# ==========================================
# 2. FUNCIÓN PARA GENERAR HEIGHTMAP
# ==========================================
def generate_terrain_heightmap(user_prompt):
    """
    Toma un texto y genera una imagen en escala de grises para UEFN.
    """
    
    # EL SECRETO DE GAIA: El "Prompt Mágico"
    # Forzamos al modelo a pensar en topografía, no en fotos reales.
    system_prefix = "grayscale topographic heightmap, overhead view, high contrast between peaks and valleys, seamless texture, unreal engine 5 landscape mask, 8k resolution, sharp edges."
    
    negative_prompt = "colors, blue sky, green grass, water, 3d objects, buildings, trees, blurry, blurry noise, text, signature"
    
    # Combinamos lo que pide el usuario con nuestras instrucciones técnicas
    final_prompt = f"{user_prompt}, {system_prefix}"

    print(f"🎨 Generando terreno para: {final_prompt}")

    # Generar imagen
    image = pipe(
        prompt=final_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=4,  # SDXL Turbo solo necesita 4 pasos (¡Rápido!)
        guidance_scale=2.0
    ).images[0]

    return image

def image_to_base64(image):
    """Convierte la imagen PIL a texto Base64 para enviarla por internet"""
    buffered = BytesIO()
    # Guardar como PNG sin compresión para mantener la calidad del heightmap
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# ==========================================
# 3. EL HANDLER (Escucha las peticiones de tu App)
# ==========================================
def handler(event):
    """
    Esta función se ejecuta cada vez que tu app hace click en "Enviar Misión".
    """
    input_data = event.get("input", {})
    
    # Obtener el texto del usuario
    prompt = input_data.get("prompt", "")
    
    if not prompt:
        return {"error": "No se recibió ningún prompt."}

    try:
        # A. Generar la imagen
        terrain_image = generate_terrain_heightmap(prompt)
        
        # B. Convertir a Base64
        img_base64 = image_to_base64(terrain_image)
        
        # C. Devolver respuesta a tu App JS
        return {
            "output": {
                "status": "success",
                "image_base64": img_base64,
                "message": f"Heightmap generado para: {prompt}"
            }
        }

    except Exception as e:
        print(f"❌ Error en GAIA: {e}")
        return {
            "error": str(e)
        }

# Iniciar el servidor de RunPod
runpod.serverless.start({"handler": handler})
