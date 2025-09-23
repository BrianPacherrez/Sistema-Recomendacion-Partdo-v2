from flask import Flask, render_template, request, redirect, url_for, session, flash, get_flashed_messages
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import psycopg2
import os

conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
cursor = conn.cursor()

app = Flask(__name__)
app.secret_key = "admin_root"  # cambia esto

# Cargar Imagenes
nombre_a_imagen = {
    "Café Bourbon": "Cafe_Bourbon.png",
    "Bourbon Coffee": "Cafe_Bourbon.png",

    "Café Geisha": "Cafe_Geisha.png",
    "Geisha Coffee": "Cafe_Geisha.png",

    "Café de Proceso Natural": "Cafe_Natural.png",
    "Natural Process Coffee": "Cafe_Natural.png",

    "Café Caturra": "Cafe_Caturra.png",
    "Caturra Coffee": "Cafe_Caturra.png",

    "Café Catuai": "Cafe_Catuai.png",
    "Catuai Coffee": "Cafe_Catuai.png",

    "Café Heirloom de Etiopia": "Cafe_Heirloom.png",
    "Ethiopian Heirloom Coffee": "Cafe_Heirloom.png",  # ajusta según traducción

    "Crema de Café": "Crema_Cafe.jpg",
    "Coffee Cream": "Crema_Cafe.jpg",

    "Jabón Exfoliante de Café": "Jabon_Cafe.jpg",
    "Exfoliating Coffee Soap": "Jabon_Cafe.jpg",
}


traducciones = {
    "es": {
        "hola": "Hola",
        "inicia_sesion": "Hola, Inicia sesión",
        "cerrar_sesion": "Cerrar sesión",
        "iniciar_sesion": "Iniciar sesión",
        "resultados": "Resultados",
        "titulo": "Recomendador de Productos de Café",
        "leer": "Leer",
        "voz": "Hablar",
        "placeholder": "Escribe tu nombre o usuario",
        "sabor_label": "Selecciona hasta 3 sabores:",
        "placeholderValue": "Selecciona sabores",
        "maxSabores": "Solo 3 valores pueden ser agregados",
        "boton": "Obtener recomendaciones",
        "encuesta": "Encuesta", 
        "sabores": ["Dulce", "Frutal", "Caramelo", "Chocolate", "Floral", "Limón", "Vino", "Suave", "Intenso", "Cítrico", "Cremoso"],
        "seleccion": "¡Genial! Has seleccionado los sabores:",
        "recomendaciones": "Recomendaciones personalizadas:",
        "historial": "Productos que ya has probado:",
    },
    "en": {
        "hola": "Hello",
        "inicia_sesion": "Hello, Log in",
        "cerrar_sesion": "Log out",
        "iniciar_sesion": "Log in",
        "resultados": "Dashboard",
        "titulo": "Coffee Product Recommender",
        "leer": "Read",
        "voz": "Speak",
        "placeholder": "Enter your name or username",
        "sabor_label": "Select up to 3 flavors:",
        "placeholderValue": "Select flavors",
        "maxSabores": "Only 3 values can be added",
        "boton": "Get Recommendations",
        "encuesta": "Survey", 
        "sabores": ["Sweet", "Fruity", "Caramel", "Chocolate", "Floral", "Lemon", "Wine", "Smooth", "Intense", "Citrus", "Creamy"],
        "seleccion": "Great! You've selected the following flavors:",
        "recomendaciones": "Personalized recommendations:",
        "historial": "Products you have already tried:",
    }
}


# Cargar productos
productos = [
    {"producto": "Café Bourbon", "categoria": "especialidades", "perfil_sabor": ["dulce", "frutal", "caramelo"]},
    {"producto": "Café Geisha", "categoria": "especialidades", "perfil_sabor": ["floral", "frutal", "limón"]},
    {"producto": "Café de Proceso Natural", "categoria": "especialidades", "perfil_sabor": ["frutal", "vino", "ácido"]},
    {"producto": "Café Caturra", "categoria": "especialidades", "perfil_sabor": ["chocolate", "nuez", "suave"]},
    {"producto": "Café Catuai", "categoria": "especialidades", "perfil_sabor": ["miel", "cítrico", "suave"]},
    {"producto": "Café Heirloom de Etiopia", "categoria": "especialidades", "perfil_sabor": ["floral", "limón", "arándano"]},
    {"producto": "Crema de Café", "categoria": "derivado", "perfil_sabor": ["dulce", "cremoso", "café"]},
    {"producto": "Jabón Exfoliante de Café", "categoria": "derivado", "perfil_sabor": ["intenso", "café", "terroso"]},
]

df_productos = pd.DataFrame(productos)
df_productos["perfil_sabor_str"] = df_productos["perfil_sabor"].apply(lambda x: " ".join(x))

# Machine Learning
vectorizer = TfidfVectorizer()
vectores_perfil = vectorizer.fit_transform(df_productos["perfil_sabor_str"])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_productos["cluster"] = kmeans.fit_predict(vectores_perfil)

# Cargar Usuarios
df_usuarios = pd.read_csv('db_usuarios_Finca_expandidos.csv', encoding='utf-8')
usuarios_csv = df_usuarios['usuario'].str.lower().unique().tolist()


@app.route('/', methods=['GET', 'POST'])
def index():
    recomendaciones = []
    historial = []
    perfil_usuario = []
    sabores_seleccionados = []
    nombre = ""
    perfil = ""

    lang = request.args.get("lang", "es")
    textos = traducciones.get(lang, traducciones["es"])

    if request.method == 'POST':
        nombre = request.form['nombre'].strip().lower()
        perfil = request.form.getlist('perfil')

        sabores_seleccionados = perfil  # tal cual lo escribió el usuario


        # Diccionario de traducción sabores
        sabores_map = {
            "es": {
                "dulce": "dulce",
                "frutal": "frutal",
                "caramelo": "caramelo",
                "chocolate": "chocolate",
                "floral": "floral",
                "limón": "limón",
                "vino": "vino",
                "suave": "suave",
                "intenso": "intenso",
                "cítrico": "cítrico",
                "cremoso": "cremoso",
            },
            "en": {
                "sweet": "dulce",
                "fruity": "frutal",
                "caramel": "caramelo",
                "chocolate": "chocolate",
                "floral": "floral",
                "lemon": "limón",
                "wine": "vino",
                "smooth": "suave",
                "intense": "intenso",
                "citrus": "cítrico",
                "creamy": "cremoso",
            }
        }

        # Convertir al español SIEMPRE antes de recomendar
        perfil_usuario = [sabores_map[lang].get(s.lower(), s) for s in perfil]


        def traducir_nombres(lista, lang):
            traducciones_productos = {
                "Café Bourbon": {"en": "Bourbon Coffee"},
                "Café Geisha": {"en": "Geisha Coffee"},
                "Café de Proceso Natural": {"en": "Natural Process Coffee"},
                "Café Caturra": {"en": "Caturra Coffee"},
                "Café Catuai": {"en": "Catuai Coffee"},
                "Café Heirloom de Etiopia": {"en": "Ethiopian Heirloom Coffee"},
                "Crema de Café": {"en": "Coffee Cream"},
                "Jabón Exfoliante de Café": {"en": "Exfoliating Coffee Soap"},
            }
            
            if lang == "es":
                return lista
            else:
                return [traducciones_productos.get(nombre, {}).get("en", nombre) for nombre in lista]

        # 🔹 Verificar si es usuario nuevo
        es_nuevo = nombre not in df_usuarios['usuario'].str.lower().unique()

        if es_nuevo:
            if not perfil_usuario or len(perfil_usuario) > 3:
                mensaje_error = "Selecciona entre 1 y 3 sabores si eres nuevo."
            else:
                recomendaciones = recomendar_nuevo_por_cluster(perfil_usuario)
        else:
            historial = df_usuarios[df_usuarios['usuario'].str.lower() == nombre]["producto"].tolist()
            recomendaciones = recomendar_por_historial(nombre)

        # Traducir productos al idioma seleccionado
        recomendaciones = traducir_nombres(recomendaciones, lang)
        historial = traducir_nombres(historial, lang)

    return render_template(
        'index.html',
        recomendaciones=recomendaciones,
        historial=historial,
        perfil_usuario=perfil_usuario,
        nombre=nombre,
        perfil=perfil,
        mensaje_error=mensaje_error if 'mensaje_error' in locals() else None,
        usuarios_csv=usuarios_csv,
        nombre_a_imagen=nombre_a_imagen,
        textos=textos,
        sabores_seleccionados=sabores_seleccionados,
        lang=lang
    )



# Recomendar por clustering
def recomendar_nuevo_por_cluster(perfil_usuario):
    perfil_str = " ".join(perfil_usuario)
    vector_usuario = vectorizer.transform([perfil_str])
    cluster_usuario = kmeans.predict(vector_usuario)[0]

    def contar_coincidencias(p_sabor):
        return len(set(p_sabor) & set(perfil_usuario))

    # Comparar con TODOS los productos (no solo los del clúster)
    candidatos = df_productos.copy()
    candidatos["coincidencias"] = candidatos["perfil_sabor"].apply(contar_coincidencias)
    recomendados = candidatos[candidatos["coincidencias"] > 0]
    recomendados = recomendados.sort_values(by="coincidencias", ascending=False)

    return recomendados["producto"].tolist()[:5]


# Usuario existente con historial + fallback
def recomendar_por_historial(usuario):
    historial = df_usuarios[df_usuarios['usuario'].str.lower() == usuario.lower()]
    if historial.empty:
        return []

    productos_usuario = historial['producto'].tolist()
    perfiles_historial = df_productos[df_productos['producto'].isin(productos_usuario)]["perfil_sabor_str"]

    if perfiles_historial.empty:
        return []

    vectores_usuario = vectorizer.transform(perfiles_historial)
    vector_promedio = np.asarray(vectores_usuario.mean(axis=0)).reshape(1, -1)

    cluster_usuario = kmeans.predict(vector_promedio)[0]
    productos_cluster = df_productos[df_productos["cluster"] == cluster_usuario]

    recomendaciones = []
    for producto in productos_cluster["producto"]:
        if producto not in productos_usuario and producto not in recomendaciones:
            recomendaciones.append(producto)
        if len(recomendaciones) >= 5:
            break


    # Fallback: si hay menos de 3 recomendaciones, buscar por perfil de sabor
    if len(recomendaciones) < 3:
        perfil_usuario = df_productos[df_productos["producto"].isin(productos_usuario)]["perfil_sabor"]
        flat_perfil = [p for sublist in perfil_usuario for p in sublist]
        top_tags = pd.Series(flat_perfil).value_counts().head(3).index.tolist()

        def contar_coincidencias(p_sabor):
            return len(set(p_sabor) & set(top_tags))

        candidatos = df_productos[~df_productos["producto"].isin(productos_usuario)].copy()
        candidatos["coincidencias"] = candidatos["perfil_sabor"].apply(contar_coincidencias)
        candidatos = candidatos[candidatos["coincidencias"] > 0].sort_values(by="coincidencias", ascending=False)

        for producto in candidatos["producto"]:
            if producto not in recomendaciones:
                recomendaciones.append(producto)
            if len(recomendaciones) >= 5:
                break

    return recomendaciones


# Configuración de conexión local
DB_HOST = "localhost"
DB_NAME = "encuesta_db"
DB_USER = "postgres"
DB_PASS = "root"

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        options="-c client_encoding=UTF8"
    )
    return conn


conn = get_db_connection()
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS votos (
    id SERIAL PRIMARY KEY,
    p1 TEXT,
    p2 TEXT,
    p3 TEXT,
    p4 TEXT,
    p5 TEXT,
    p6 TEXT,
    p7 TEXT,
    p8 TEXT,
    p9 TEXT,
    p10 TEXT,
    p11 TEXT,
    p12 TEXT,
    p13 TEXT,
    p14 TEXT,
    p15 TEXT                
)
""")
conn.commit()
cur.close()
conn.close()


@app.route("/encuesta", methods=["GET", "POST"])
def encuesta():
    if request.method == "POST":
        respuestas = [request.form.get(f"p{i}") for i in range(1, 16)]

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO votos (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                               p11, p12, p13, p14, p15)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s)
        """, respuestas)

        conn.commit()
        cursor.close()
        conn.close()

        # limpiar flashes anteriores
        get_flashed_messages()

        flash("✅ Encuesta grabada con éxito", "success")
        return redirect(url_for("index"))

    return render_template("encuesta.html")


@app.route("/resultados")
def resultados():
    if session.get("rol") != "admin":
        flash("Acceso denegado. Debes iniciar sesión.", "danger")
        return redirect(url_for("login"))
    
    conn = get_db_connection()
    cur = conn.cursor()

    # Pregunta 1
    cur.execute("SELECT p1, COUNT(*) FROM votos GROUP BY p1;")
    resultados_p1 = cur.fetchall()
    labels = [fila[0] for fila in resultados_p1]
    values = [fila[1] for fila in resultados_p1]

    # Pregunta 2
    cur.execute("SELECT p2, COUNT(*) FROM votos GROUP BY p2;")
    resultados_p2 = cur.fetchall()
    labels_p2 = [fila[0] for fila in resultados_p2]
    values_p2 = [fila[1] for fila in resultados_p2]

    # Pregunta 3
    cur.execute("SELECT p3, COUNT(*) FROM votos GROUP BY p3;")
    resultados_p3 = cur.fetchall()
    labels_p3 = [fila[0] for fila in resultados_p3]
    values_p3 = [fila[1] for fila in resultados_p3]

    # Pregunta 4
    cur.execute("SELECT p4, COUNT(*) FROM votos GROUP BY p4;")
    resultados_p4 = cur.fetchall()
    labels_p4 = [fila[0] for fila in resultados_p4]
    values_p4 = [fila[1] for fila in resultados_p4]

    # Pregunta 5
    cur.execute("SELECT p5, COUNT(*) FROM votos GROUP BY p5;")
    resultados_p5 = cur.fetchall()
    labels_p5 = [fila[0] for fila in resultados_p5]
    values_p5 = [fila[1] for fila in resultados_p5]

    # Pregunta 6
    cur.execute("SELECT p6, COUNT(*) FROM votos GROUP BY p6;")
    resultados_p6 = cur.fetchall()
    labels_p6 = [fila[0] for fila in resultados_p6]
    values_p6 = [fila[1] for fila in resultados_p6]

    # Pregunta 7
    cur.execute("SELECT p7, COUNT(*) FROM votos GROUP BY p7;")
    resultados_p7 = cur.fetchall()
    labels_p7 = [fila[0] for fila in resultados_p7]
    values_p7 = [fila[1] for fila in resultados_p7]

    # Pregunta 8
    cur.execute("SELECT p8, COUNT(*) FROM votos GROUP BY p8;")
    resultados_p8 = cur.fetchall()
    labels_p8 = [fila[0] for fila in resultados_p8]
    values_p8 = [fila[1] for fila in resultados_p8]

    # Pregunta 9
    cur.execute("SELECT p9, COUNT(*) FROM votos GROUP BY p9;")
    resultados_p9 = cur.fetchall()
    labels_p9 = [fila[0] for fila in resultados_p9]
    values_p9 = [fila[1] for fila in resultados_p9]

    # Pregunta 10
    cur.execute("SELECT p10, COUNT(*) FROM votos GROUP BY p10;")
    resultados_p10 = cur.fetchall()
    labels_p10 = [fila[0] for fila in resultados_p10]
    values_p10 = [fila[1] for fila in resultados_p10]

    # Pregunta 11
    cur.execute("SELECT p11, COUNT(*) FROM votos GROUP BY p11;")
    resultados_p11 = cur.fetchall()
    labels_p11 = [fila[0] for fila in resultados_p11]
    values_p11 = [fila[1] for fila in resultados_p11]

    # Pregunta 12
    cur.execute("SELECT p12, COUNT(*) FROM votos GROUP BY p12;")
    resultados_p12 = cur.fetchall()
    labels_p12 = [fila[0] for fila in resultados_p12]
    values_p12 = [fila[1] for fila in resultados_p12]

    # Pregunta 13
    cur.execute("SELECT p13, COUNT(*) FROM votos GROUP BY p13;")
    resultados_p13 = cur.fetchall()
    labels_p13 = [fila[0] for fila in resultados_p13]
    values_p13 = [fila[1] for fila in resultados_p13]

    # Pregunta 14
    cur.execute("SELECT p14, COUNT(*) FROM votos GROUP BY p14;")
    resultados_p14 = cur.fetchall()
    labels_p14 = [fila[0] for fila in resultados_p14]
    values_p14 = [fila[1] for fila in resultados_p14]

    # Pregunta 15
    cur.execute("SELECT p15, COUNT(*) FROM votos GROUP BY p15;")
    resultados_p15 = cur.fetchall()
    labels_p15 = [fila[0] for fila in resultados_p15]
    values_p15 = [fila[1] for fila in resultados_p15]

    cur.close()
    conn.close()

    return render_template(
        "resultados.html",
        labels=labels, values=values,
        labels_p2=labels_p2, values_p2=values_p2,
        labels_p3=labels_p3, values_p3=values_p3,
        labels_p4=labels_p4, values_p4=values_p4,
        labels_p5=labels_p5, values_p5=values_p5,
        labels_p6=labels_p6, values_p6=values_p6,
        labels_p7=labels_p7, values_p7=values_p7,
        labels_p8=labels_p8, values_p8=values_p8,
        labels_p9=labels_p9, values_p9=values_p9,
        labels_p10=labels_p10, values_p10=values_p10,
        labels_p11=labels_p11, values_p11=values_p11,
        labels_p12=labels_p12, values_p12=values_p12,
        labels_p13=labels_p13, values_p13=values_p13,
        labels_p14=labels_p14, values_p14=values_p14,
        labels_p15=labels_p15, values_p15=values_p15
    )


@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        # limpiar mensajes anteriores
        get_flashed_messages()

        if username == "admin" and password == "12345":
            session["rol"] = "admin"
            session["usuario"] = username
            flash("👋 Bienvenido, admin", "success")
            return redirect(url_for("index"))
        else:
            flash("❌ Credenciales incorrectas", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("🔒 Sesión cerrada", "info")
    return redirect(url_for("index"))



if __name__ == "__main__":
    app.run(debug=True)