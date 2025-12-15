# 🧼 Estandarizador de Datos con IA (RAG Architecture)

Herramienta de normalización semántica que utiliza Inteligencia Artificial para limpiar y estandarizar registros de datos empresariales (Hardware y Empresas), mapeando entradas "sucias" hacia un Catálogo Maestro Oficial.

## 📋 Requisitos Previos

* **Sistema Operativo:** Windows, macOS o Linux.
* **Python:** Versión 3.9 o superior.
* **Git:** Para clonar el repositorio.

---

## 🛠️ Instalación y Configuración Local

Sigue estos pasos secuenciales para levantar el proyecto en tu máquina.

### 1. Clonar el Repositorio
Abre tu terminal o línea de comandos y ejecuta:

```bash
git clone [https://github.com/LIMPIEZA-DE-DATOS-PERU.git](https://github.com/LIMPIEZA-DE-DATOS-PERU.git)
cd LIMPIEZA-DE-DATOS-PERU

2. Crear Entorno Virtual (Recomendado)
Es buena práctica aislar las dependencias del proyecto.

En Windows:

Bash

python -m venv venv
venv\Scripts\activate

En macOS/Linux:

Bash

python3 -m venv venv
source venv/bin/activate
3. Instalar Dependencias
Instala las librerías necesarias (Gradio, Sentence-Transformers, Pandas, etc.):

Bash

pip install -r requirements.txt
🚀 Ejecución del Proyecto
El sistema funciona en dos fases: Creación del Índice y Ejecución de la Interfaz.

Fase 1: Generar el "Cerebro" (Indexación)
Antes de usar la aplicación, debes crear el archivo vectorial (.pkl) que contiene la inteligencia del catálogo.

Asegúrate de que el archivo catalogo_maestro.csv esté en la carpeta raíz.

Ejecuta el script de generación:

Bash

python generar_indice.py
Nota: Esto descargará el modelo de IA la primera vez y generará el archivo cerebro_expandido.pkl. Verás un mensaje de "ÉXITO" al finalizar.

Fase 2: Iniciar la Aplicación Web
Una vez generado el cerebro, levanta la interfaz gráfica:

Bash

python app.py
La terminal mostrará una URL local. Abre tu navegador y ve a: http://127.0.0.1:7860

📂 Estructura de Archivos
app.py: Código principal de la interfaz web (Gradio).

generar_indice.py: Script para procesar el CSV y crear los embeddings.

catalogo_maestro.csv: Base de datos fuente (Columnas: Variante_Busqueda, Nombre_Oficial).

cerebro_expandido.pkl: Archivo binario generado (Vector Store).

requirements.txt: Lista de dependencias.

⚙️ Cómo Actualizar los Datos
Si deseas agregar nuevos productos o empresas al sistema:

Edita el archivo catalogo_maestro.csv.

Agrega las nuevas filas respetando el formato CSV.

Vuelve a ejecutar python generar_indice.py para actualizar el cerebro.

Reinicia la aplicación app.py.

Autor: [PAZ LOAIZA ARTURO JOSUE, QUISPE BRAVO KELVIN RONNY]
