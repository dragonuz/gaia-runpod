# Usamos la imagen oficial de RunPod (YA tiene Python y CUDA instalados y en caché)
# Esto hace que el inicio sea 10x más rápido.
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Directorio de trabajo
WORKDIR /

# Copiar requirements primero (para aprovechar caché de Docker)
COPY requirements.txt .

# Instalar dependencias (sin caché de pip para ahorrar espacio)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código del handler
COPY . .

# Comando de arranque
CMD [ "python", "-u", "handler.py" ]
