import gradio as gr
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import pickle
import os

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
MODEL_CACHE = None

def cargar_modelo():
    global MODEL_CACHE
    if MODEL_CACHE is None:
        try: MODEL_CACHE = SentenceTransformer(MODEL_NAME)
        except: MODEL_CACHE = SentenceTransformer('all-MiniLM-L6-v2')
    return MODEL_CACHE

def leer_dataset_robusto(file_obj):
    if file_obj is None: return None
    if isinstance(file_obj, str): file_path = file_obj
    elif hasattr(file_obj, 'name'): file_path = file_obj.name
    else: file_path = str(file_obj)
    
    ext = os.path.splitext(file_path)[-1].lower()
    try:
        if ext == '.csv':
            try: return pd.read_csv(file_path)
            except: return pd.read_csv(file_path, sep=';', encoding='latin1')
        elif ext in ['.xlsx', '.xls']: return pd.read_excel(file_path)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
            return pd.DataFrame(lines, columns=['Variante_Busqueda'])
        else: return None
    except: return None

# --- NUEVA LÓGICA DE ÍNDICE (DICCIONARIO) ---
def crear_indice_maestro(archivo_maestro):
    df = leer_dataset_robusto(archivo_maestro)
    if df is None: return None, "❌ Error leyendo archivo."

    # Detectar columnas
    # Esperamos columna 1: Lo que se busca (ej: BCP, Banco Credito)
    # Esperamos columna 2: El oficial (ej: BANCO DE CREDITO DEL PERU)
    cols = df.columns
    if len(cols) < 2:
        col_busqueda = cols[0]
        col_oficial = cols[0] # Si solo hay una, oficial y búsqueda son lo mismo
    else:
        col_busqueda = cols[0] # Variante_Busqueda
        col_oficial = cols[1]  # Nombre_Oficial

    # Crear listas paralelas
    lista_busqueda = df[col_busqueda].dropna().astype(str).tolist()
    lista_oficial = df[col_oficial].dropna().astype(str).tolist()
    
    # Vectorizar solo las variantes de búsqueda
    model = cargar_modelo()
    print(f"🧠 Indexando {len(lista_busqueda)} variantes...")
    embeddings = model.encode(lista_busqueda, convert_to_tensor=True)
    
    datos = {
        "variantes": lista_busqueda,  # Lo que se compara
        "oficiales": lista_oficial,   # Lo que se devuelve
        "embeddings": embeddings
    }
    
    with open("cerebro_expandido.pkl", "wb") as f: pickle.dump(datos, f)
    return "cerebro_expandido.pkl", f"✅ Índice Expandido creado con {len(lista_busqueda)} variantes."

# --- NUEVA LÓGICA DE LIMPIEZA ---
def limpiar_con_indice(archivo_sucio, col_sucia, archivo_indice, umbral):
    if not archivo_sucio or not archivo_indice: return None, None, "⚠️ Faltan archivos."

    try:
        pkl_path = archivo_indice if isinstance(archivo_indice, str) else archivo_indice.name
        with open(pkl_path, "rb") as f: data = pickle.load(f)
        
        # Recuperamos listas paralelas
        variantes_maestras = data["variantes"]
        oficiales_maestros = data["oficiales"]
        embeddings_maestro = data["embeddings"]
    except: return None, None, "❌ Error en .pkl"

    df_sucio = leer_dataset_robusto(archivo_sucio)
    if df_sucio is None: return None, None, "❌ Error en archivo sucio."
    
    if col_sucia not in df_sucio.columns:
        if 'Variante_Busqueda' in df_sucio.columns: col_sucia = 'Variante_Busqueda'
        else: col_sucia = df_sucio.columns[0]

    model = cargar_modelo()
    lista_sucia = df_sucio[col_sucia].dropna().astype(str).unique().tolist()
    embeddings_sucio = model.encode(lista_sucia, convert_to_tensor=True)

    hits = util.semantic_search(embeddings_sucio, embeddings_maestro, top_k=1)
    
    mapa = {}
    reporte = []

    for i, hit in enumerate(hits):
        sucio = lista_sucia[i]
        match = hit[0]
        score = match['score']
        
        if score >= umbral:
            idx = match['corpus_id']
            # AQUÍ ESTÁ LA MAGIA:
            # Encontramos coincidencia con "BCP" (idx 5), pero devolvemos el oficial en la pos 5 "BANCO DE CREDITO..."
            nombre_final = oficiales_maestros[idx] 
            variante_encontrada = variantes_maestras[idx]
            
            mapa[sucio] = nombre_final
            
            if sucio != nombre_final:
                reporte.append({
                    "Original": sucio, 
                    "Match con Variante": variante_encontrada,
                    "Resultado Oficial": nombre_final,
                    "Confianza": f"{score:.2f}"
                })
        else:
            mapa[sucio] = sucio

    df_sucio[f"{col_sucia}_NORMALIZADO"] = df_sucio[col_sucia].astype(str).map(mapa)
    
    import tempfile
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx").name
    df_sucio.to_excel(out, index=False)
    
    return out, pd.DataFrame(reporte), f"✅ Procesado. {len(reporte)} coincidencias."

# --- INTERFAZ ---
with gr.Blocks(title="Limpiador Expandido") as demo:
    gr.Markdown("# 🚀 Limpiador Universal (Estrategia Diccionario)")
    gr.Markdown("Usa un catálogo con Múltiples Variantes para encontrar el Nombre Oficial.")
    
    with gr.Tabs():
        with gr.TabItem("1. Generar Cerebro"):
            gr.Markdown("Sube `catalogo_expandido.csv` (Columna 1: Variantes, Columna 2: Oficial)")
            f_m = gr.File(label="Catálogo Expandido")
            btn_gen = gr.Button("💾 CREAR CEREBRO", variant="primary")
            out_pkl = gr.File(label="Descargar .pkl")
            out_log = gr.Textbox(label="Estado")
            btn_gen.click(crear_indice_maestro, [f_m], [out_pkl, out_log])

        with gr.TabItem("2. Limpiar"):
            gr.Markdown("Sube el `.pkl` y tus datos sucios.")
            with gr.Row():
                f_d = gr.File(label="Datos Sucios")
                f_k = gr.File(label="Cerebro (.pkl)")
            
            t_col = gr.Textbox(label="Columna Sucia", value="Producto_Sucio")
            sl = gr.Slider(0.5, 0.99, value=0.75, label="Confianza (Sugerido: 0.75)")
            
            btn_run = gr.Button("✨ LIMPIAR", variant="primary")
            out_f = gr.File(label="Resultado")
            out_t = gr.Dataframe(label="Reporte Detallado")
            out_s = gr.Textbox(label="Estado")
            
            btn_run.click(limpiar_con_indice, [f_d, t_col, f_k, sl], [out_f, out_t, out_s])

if __name__ == "__main__":
    demo.launch()