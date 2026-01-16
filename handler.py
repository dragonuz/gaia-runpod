import runpod
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from PIL import Image, ImageOps
import io
import base64
import numpy as np
import cv2

print("🏗️ INICIANDO GAIA 2.0: MODO PISTA DE CARRERAS...")

# 1. Cargar ControlNet (Para entender la forma de la pista)
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)

# 2. Cargar VAE (Mejor calidad de imagen)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

# 3. Cargar el Modelo SDXL
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to("cuda")

print("✅ GAIA ONLINE: Listo para recibir planos.")

def process_image_for_controlnet(image):
    """Prepara la imagen para que la IA entienda los bordes"""
    image = image.convert("RGB")
    image_np = np.array(image)
    # Detectar bordes de la pista
    image_np = cv2.Canny(image_np, 100, 200)
    image_np = image_np[:, :, None]
    image_np = np.concatenate([image_np, image_np, image_np], axis=2)
    return Image.fromarray(image_np)

def handler(event):
    input_data = event.get("input", {})
    
    # INPUTS
    drawing_base64 = input_data.get("image_base64") # El dibujo de Google
    user_prompt = input_data.get("prompt", "rocky mountains") # El entorno
    
    if not drawing_base64:
        return {"error": "Falta el dibujo de la pista (image_base64)"}

    try:
        # 1. Decodificar el dibujo
        image_data = base64.b64decode(drawing_base64)
        original_drawing = Image.open(io.BytesIO(image_data)).convert("L")
        
        # Redimensionar a 1024 para la IA
        control_image = original_drawing.resize((1024, 1024))
        canny_image = process_image_for_controlnet(control_image)

        # 2. Generar el Terreno con IA (Montañas alrededor de la pista)
        # Forzamos el prompt para que sea un Heightmap
        tech_prompt = f"grayscale heightmap, top-down aerial view, {user_prompt}, high contrast, smooth terrain, 8k"
        neg_prompt = "colors, trees, water, buildings, perspective, isometric, noise"

        print(f"🎨 Generando terreno para: {user_prompt}")
        
        generated_terrain = pipe(
            prompt=tech_prompt,
            negative_prompt=neg_prompt,
            image=canny_image,
            controlnet_conditioning_scale=0.6, # Seguir la forma de la pista al 60%
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

        # 3. LA ESTOCADA FINAL: APLANADO MATEMÁTICO
        # Donde el dibujo original es NEGRO (Pista), forzamos un gris plano perfecto.
        # Donde es BLANCO (Fondo), dejamos las montañas de la IA.
        
        # Convertir a arrays matemáticos
        terrain_np = np.array(generated_terrain.convert("L").resize((1009, 1009))) # Tamaño UEFN
        drawing_np = np.array(original_drawing.resize((1009, 1009)))

        # Crear el mapa final de 16 bits (0 a 65535)
        final_heightmap = np.zeros((1009, 1009), dtype=np.uint16)
        
        # Escalar el terreno de la IA a 16 bits
        terrain_16bit = terrain_np.astype(np.uint16) * 256
        
        # Definir altura de la pista (Gris medio perfecto)
        TRACK_HEIGHT = 32768 
        
        # Aplicar la lógica: Si en el dibujo hay pista (negro/oscuro), ponlo plano.
        # Si hay fondo (blanco/claro), pon el terreno de la IA.
        # (Ajusta el < 100 según qué tan negro sea el dibujo de Google)
        is_track = drawing_np < 100 
        
        final_heightmap[:] = terrain_16bit[:] # Copiar terreno base
        final_heightmap[is_track] = TRACK_HEIGHT # Aplanar pista

        # 4. Guardar y Enviar
        out_img = Image.fromarray(final_heightmap, mode='I;16')
        buffered = io.BytesIO()
        out_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"status": "success", "image_base64": img_str}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
