import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import os
import tempfile

# --- CONFIGURACIÓN ESTÁTICA ---
# El nombre EXACTO del archivo que subiste a "Files" en Hugging Face
ARCHIVO_CEREBRO = "cerebro_expandido.pkl" 
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# --- VARIABLES GLOBALES (Lazy Loading) ---
MODEL_CACHE = None
CEREBRO_CACHE = None

# --- CARGA DEL MODELO (Solo cuando se usa) ---
def cargar_modelo():
    global MODEL_CACHE
    if MODEL_CACHE is None:
        print("⏳ Cargando modelo IA...")
        try:
            MODEL_CACHE = SentenceTransformer(MODEL_NAME)
            print("✅ Modelo cargado.")
        except:
            MODEL_CACHE = SentenceTransformer('all-MiniLM-L6-v2')
    return MODEL_CACHE

# --- CARGA DEL CEREBRO (.PKL) ---
def cargar_cerebro():
    global CEREBRO_CACHE
    
    # Si ya está en memoria, no lo leemos de nuevo
    if CEREBRO_CACHE is not None:
        return CEREBRO_CACHE

    print(f"🧠 Buscando cerebro: {ARCHIVO_CEREBRO}...")
    
    if not os.path.exists(ARCHIVO_CEREBRO):
        raise FileNotFoundError(f"❌ ERROR CRÍTICO: No encontré el archivo '{ARCHIVO_CEREBRO}' en el repositorio. Por favor súbelo en la pestaña Files.")
    
    try:
        with open(ARCHIVO_CEREBRO, "rb") as f:
            datos = pickle.load(f)
        
        # Validamos que sea el cerebro nuevo (con variantes)
        if "variantes" not in datos or "oficiales" not in datos:
             raise ValueError("El archivo .pkl es antiguo. Necesitas generar el 'Cerebro Expandido' de nuevo.")
             
        CEREBRO_CACHE = datos
        print("✅ Cerebro cargado en memoria RAM.")
        return CEREBRO_CACHE
    except Exception as e:
        raise Exception(f"Error cargando el cerebro: {e}")

# --- LECTOR DE ARCHIVOS ---
def leer_archivo(file_obj):
    if file_obj is None: return None
    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    ext = os.path.splitext(file_path)[-1].lower()
    
    try:
        if ext == '.csv':
            try: return pd.read_csv(file_path)
            except: return pd.read_csv(file_path, sep=';', encoding='latin1')
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        else: return None
    except: return None

# --- LÓGICA PRINCIPAL ---
def limpiar_datos(archivo_sucio, col_sucia, umbral):
    # 1. Cargar Recursos
    try:
        datos_cerebro = cargar_cerebro() # Carga el pkl
        model = cargar_modelo()          # Carga la IA
    except Exception as e:
        return None, None, f"❌ Error de Sistema: {str(e)}"

    # 2. Leer Usuario
    df = leer_archivo(archivo_sucio)
    if df is None: return None, None, "❌ Error leyendo tu archivo. Usa Excel o CSV."

    if col_sucia not in df.columns:
        col_sucia = df.columns[0] # Fallback a la primera columna
        aviso_col = f"(Usé la columna '{col_sucia}')"
    else:
        aviso_col = ""

    # 3. Vectorizar lo sucio
    lista_sucia = df[col_sucia].dropna().astype(str).unique().tolist()
    print(f"🔎 Analizando {len(lista_sucia)} términos únicos...")
    embeddings_sucio = model.encode(lista_sucia, convert_to_tensor=True)

    # 4. Búsqueda RAG (Contra el Cerebro Cargado)
    hits = util.semantic_search(embeddings_sucio, datos_cerebro["embeddings"], top_k=1)
    
    mapa = {}
    reporte = []

    for i, hit in enumerate(hits):
        sucio = lista_sucia[i]
        match = hit[0]
        score = match['score']
        
        if score >= umbral:
            idx_maestro = match['corpus_id']
            
            # MAGIA: La variante coincidió, pero devolvemos el OFICIAL
            variante_match = datos_cerebro["variantes"][idx_maestro]
            nombre_oficial = datos_cerebro["oficiales"][idx_maestro]
            
            mapa[sucio] = nombre_oficial
            
            if sucio != nombre_oficial:
                reporte.append({
                    "Original": sucio,
                    "Encontrado en Catálogo": variante_match,
                    "Estandarizado A": nombre_oficial,
                    "Confianza": f"{score:.2f}"
                })
        else:
            mapa[sucio] = sucio # No se cambia si no es seguro

    # 5. Guardar
    col_nueva = f"{col_sucia}_LIMPIO"
    df[col_nueva] = df[col_sucia].astype(str).map(mapa)
    
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
    df.to_excel(out_file, index=False)
    
    mensaje_final = f"✅ ¡Listo! Se estandarizaron {len(reporte)} datos. {aviso_col}"
    return out_file, pd.DataFrame(reporte), mensaje_final

# --- INTERFAZ USUARIO FINAL ---
with gr.Blocks(title="Estandarizador Pro") as demo:
    gr.Markdown("# 🧼 Estandarizador de Datos (Perú + Tech)")
    gr.Markdown("Sube tu archivo desordenado y la IA lo limpiará usando el Catálogo Maestro Oficial.")
    
    with gr.Row():
        with gr.Column(scale=1):
            f_input = gr.File(label="📂 1. Sube tu Excel o CSV")
            txt_col = gr.Textbox(label="📝 2. Nombre de la Columna a limpiar", value="Producto_Sucio", placeholder="Ej: Cliente")
            
            with gr.Accordion("⚙️ Configuración Avanzada", open=False):
                # Default 0.75 es ideal para el cerebro expandido
                sl_conf = gr.Slider(0.5, 0.99, value=0.75, label="Umbral de Confianza")
                gr.Info("Si el cerebro (.pkl) está en el servidor, esto funcionará automáticamente.")

            btn_run = gr.Button("🚀 LIMPIAR DATOS AHORA", variant="primary")
        
        with gr.Column(scale=2):
            lbl_status = gr.Textbox(label="Estado del Proceso")
            f_output = gr.File(label="📥 3. Descargar Resultado")
            tbl_report = gr.Dataframe(label="📊 Resumen de Cambios (Muestra)")

    btn_run.click(limpiar_datos, [f_input, txt_col, sl_conf], [f_output, tbl_report, lbl_status])

if __name__ == "__main__":
    demo.launch()
