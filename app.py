from flask import Flask, render_template, request, redirect, url_for, session, flash, get_flashed_messages, jsonify
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

# Flask
app.secret_key = os.environ.get("SECRET_KEY")

# PostgreSQL
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")

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
        "detener": "Detener",
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
        "asistente": "Asistente Virtual del Café",
        "chat_emisor": "Asistente",
        "chat_respuesta": "¡Hola! Estoy aquí para ayudarte 😊 Pregúntame sobre cafés, sabores o lo que desees conocer ☕",
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
        "detener": "Stop",
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
        "asistente": "Virtual Coffee Assistant",
        "chat_emisor": "Assistant",
        "chat_respuesta": "Hello! I'm here to help you 😊 Ask me about coffees, flavors, or anything else you want to know ☕",
    },
    "br": {
        "hola": "Olá",
        "inicia_sesion": "Olá, faça login",
        "cerrar_sesion": "Sair",
        "iniciar_sesion": "Entrar",
        "resultados": "Resultados",
        "titulo": "Recomendador de Produtos de Café",
        "leer": "Ler",
        "voz": "Falar",
        "detener": "Parar",
        "placeholder": "Digite seu nome ou nome de usuário",
        "sabor_label": "Selecione até 3 sabores:",
        "placeholderValue": "Selecionar sabores",
        "maxSabores": "Apenas 3 valores podem ser adicionados",
        "boton": "Receber recomendações",
        "encuesta": "Enquete", 
        "sabores": ["Doce", "Frutal", "Caramelo", "Chocolate", "Floral", "Limão", "Vinho", "Suave", "Intenso", "Cítrico", "Cremoso"],
        "seleccion": "Ótimo! Vocês selecionou os sabores:",
        "recomendaciones": "Recomendações personalizadas:",
        "historial": "Produtos que vocês ja provou:",
        "asistente": "Assistente Virtual de Café",
        "chat_emisor": "Assistente",
        "chat_respuesta": "Olá! Estou aqui para te ajudar 😊 Pergunte-me sobre cafés, sabores ou qualquer outra coisa que deseje saber ☕",
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


        # Diccionario para traducir sabores
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
            },
            "br": {
                "doce": "dulce",
                "frutal": "frutal",
                "caramelo": "caramelo",
                "chocolate": "chocolate",
                "floral": "floral",
                "limão": "limão",
                "vinho": "vino",
                "suave": "suave",
                "intenso": "intenso",
                "cítrico": "cítrico",
                "cremoso": "cremoso",
            } 
        }

        # Convertir en español siempre antes de recomendar
        perfil_usuario = [sabores_map[lang].get(s.lower(), s) for s in perfil]


        def traducir_nombres(lista, lang):
            traducciones_productos = {
                "Café Bourbon": {"en": "Bourbon Coffee", "br": "Café Bourbon"},
                "Café Geisha": {"en": "Geisha Coffee", "br": "Café Geisha"},
                "Café de Proceso Natural": {"en": "Natural Process Coffee", "br": "Café Processado Naturalmente"},
                "Café Caturra": {"en": "Caturra Coffee", "br": "Café Caturra"},
                "Café Catuai": {"en": "Catuai Coffee", "br": "Café Catuai"},
                "Café Heirloom de Etiopia": {"en": "Ethiopian Heirloom Coffee", "br": "Café Heirloom de Etiopia"},
                "Crema de Café": {"en": "Coffee Cream", "br": "Creme de Café"},
                "Jabón Exfoliante de Café": {"en": "Exfoliating Coffee Soap", "br": "Sabonete Esfoliante de Café"},
            }
            
            if lang == "es":
                return lista
            else:
                return [traducciones_productos.get(nombre, {}).get("en", nombre) for nombre in lista]

        # Verificar si es usuario nuevo
        es_nuevo = nombre not in df_usuarios['usuario'].str.lower().unique()

        if es_nuevo:
            if not perfil_usuario or len(perfil_usuario) > 3:
                mensaje_error = "Selecciona entre 1 y 3 sabores si eres nuevo."
            else:
                recomendaciones = recomendar_nuevo_por_cluster(perfil_usuario)
        else:
            historial = df_usuarios[df_usuarios['usuario'].str.lower() == nombre]["producto"].tolist()
            recomendaciones = recomendar_por_historial(nombre)

        # Traducir los productos según el idioma seleccionado
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
# DB_HOST = "localhost"
# DB_NAME = "encuesta_db"
# DB_USER = "postgres"
# DB_PASS = "root"

# def get_db_connection():
#     conn = psycopg2.connect(
#         host=DB_HOST,
#         database=DB_NAME,
#         user=DB_USER,
#         password=DB_PASS,
#         options="-c client_encoding=UTF8"
#     )
#     return conn

def get_db_connection():
    conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
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
    p15 TEXT,
    correo TEXT UNIQUE
)
""")
conn.commit()
cur.close()
conn.close()


from psycopg2 import errors

@app.route("/encuesta", methods=["GET", "POST"])
def encuesta():
    if request.method == "POST":
        correo = request.form["correo"]
        respuestas = [request.form.get(f"p{i}") for i in range(1, 16)]

        conn = get_db_connection()
        cursor = conn.cursor()

        # Verificar si ya existe ese correo
        cursor.execute("SELECT 1 FROM votos WHERE correo = %s", (correo,))
        existe = cursor.fetchone()

        if existe:
            cursor.close()
            conn.close()
            flash("⚠️ Este correo ya ha participado en la encuesta.", "warning")
            return redirect(url_for("index"))

        # Insertar nuevo voto
        cursor.execute("""
            INSERT INTO votos (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                               p11, p12, p13, p14, p15, correo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s)
        """, (*respuestas, correo))

        conn.commit()
        cursor.close()
        conn.close()

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

        if username == "admin" and password == "srpartdoroot":
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


@app.route("/submit", methods=["POST"])
def submit():
    correo = request.form["correo"]
    respuestas = [request.form.get(f"p{i}") for i in range(1, 16)]

    conn = get_db_connection()
    cursor = conn.cursor()

    # Verificar si ya existe ese correo
    cursor.execute("SELECT 1 FROM votos WHERE correo = %s", (correo,))
    existe = cursor.fetchone()

    if existe:
        conn.close()
        return "⚠️ Este correo ya ha participado en la encuesta.", 400

    # Insertar nuevo voto
    cursor.execute("""
        INSERT INTO votos (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10,
                           p11, p12, p13, p14, p15, correo)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s)
    """, (*respuestas, correo))

    conn.commit()
    conn.close()

    return "✅ Gracias por participar."


@app.route("/dashboard")
def dashboard():
    conn = get_db_connection()
    cur = conn.cursor()

    # Total de encuestas
    cur.execute("SELECT COUNT(*) FROM votos")
    total_encuestas = cur.fetchone()[0]

    # Correos de los encuestados
    cur.execute("SELECT id, correo, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15 FROM votos")
    rows = cur.fetchall()

    datos = [{"id": row[0], "correo": row[1], "p1": row[2], "p2": row[3], "p3": row[4], "p4": row[5]
              , "p5": row[6], "p6": row[7], "p7": row[8], "p8": row[9], "p9": row[10], "p10": row[11]
              , "p11": row[12], "p12": row[13], "p13": row[14], "p14": row[15], "p15": row[16]} for row in rows]


    # Cantidad de respuestas
    cur.execute("""
    SELECT
        -- Positivas
        (
            (SELECT COUNT(*) FROM votos WHERE p1 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p2 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p3 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p4 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p5 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p6 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p7 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p8 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p9 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p10 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p11 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p12 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p13 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p14 IN ('De acuerdo', 'Totalmente de acuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p15 IN ('De acuerdo', 'Totalmente de acuerdo'))
        ) AS positivas,

        -- Neutras
        (
            (SELECT COUNT(*) FROM votos WHERE p1 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p2 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p3 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p4 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p5 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p6 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p7 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p8 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p9 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p10 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p11 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p12 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p13 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p14 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p15 IN ('Ligeramente de acuerdo', 'Ligeramente en desacuerdo'))
        ) AS neutras,

        -- Negativas
        (
            (SELECT COUNT(*) FROM votos WHERE p1 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p2 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p3 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p4 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p5 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p6 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p7 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p8 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p9 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p10 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p11 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p12 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p13 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p14 IN ('En desacuerdo', 'Totalmente en desacuerdo')) +
            (SELECT COUNT(*) FROM votos WHERE p15 IN ('En desacuerdo', 'Totalmente en desacuerdo'))
        ) AS negativas;
        """)

    positivas, neutras, negativas  = cur.fetchone()

    return render_template(
        "dashboard.html",
        total_encuestas=total_encuestas,
        positivas=positivas,
        neutras=neutras,
        negativas=negativas,
        datos=datos
    )





def extraer_sabores(texto):
    sabores_validos = [
        "dulce","frutal","caramelo","chocolate","floral",
        "limón","vino","suave","intenso","cítrico","cremoso"
    ]
    texto = texto.lower()
    return [s for s in sabores_validos if s in texto]



# from faster_whisper import WhisperModel
import tempfile
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# modelo_whisper = WhisperModel(
#     "tiny",
#     device="cpu",
#     compute_type="int8"
# )

# whisper_model = None

# def get_whisper_model():
#     global whisper_model
#     if whisper_model is None:
#         whisper_model = WhisperModel(
#             "tiny",
#             device="cpu",
#             compute_type="int8"
#         )
#     return whisper_model

@app.route("/chat_audio", methods=["POST"])
def chat_audio():
    audio = request.files["audio"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio.save(tmp.name)
        path = tmp.name

    # faster-whisper a texto
    # model = get_whisper_model()
    # segments, info = model.transcribe(
    #     path,
    #     language="es",
    #     beam_size=5,
    #     vad_filter=True
    # )

    # 🎙️ Speech-to-Text OpenAI
    with open(path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-transcribe",
            language="es"
        )
    
    # segments, info = modelo_whisper.transcribe(
    #     path,
    #     language="es"
    # )

    # texto_usuario = " ".join([segment.text for segment in segments])
    texto_usuario = transcript.text.strip()

    os.remove(path)

    if not texto_usuario:
        return jsonify({
            "texto_usuario": "",
            "respuesta": "No logré escucharte bien. ¿Podrías repetirlo?"
        })

    from utils.prompts import cargar_prompt

    contexto = cargar_prompt("cafe_recomendacion.txt").format(
        # perfil_usuario=", ".join(texto_usuario),
        perfil_usuario=texto_usuario,
        productos=df_productos[['producto','perfil_sabor']].to_string(index=False)
    )

    # GPT
    respuesta = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": contexto},
            {"role": "user", "content": texto_usuario}
        ]
    )

    # texto_respuesta = respuesta.choices[0].message.content

    # return {
    #     "texto_usuario": texto_usuario,
    #     "respuesta": texto_respuesta
    # }

    return jsonify({
        "texto_usuario": texto_usuario,
        "respuesta": respuesta.choices[0].message.content
    })

if __name__ == "__main__":

    app.run(debug=True)


















