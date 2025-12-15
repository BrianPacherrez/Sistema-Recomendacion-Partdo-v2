from pathlib import Path

def cargar_prompt(nombre_archivo):
    ruta = Path(__file__).parent.parent / "prompts" / nombre_archivo
    with open(ruta, "r", encoding="utf-8") as f:
        return f.read()
