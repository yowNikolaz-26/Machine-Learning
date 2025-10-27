import random

# Parámetros de la encuesta
num_respuestas = 1000  # Número de encuestados
num_features = 200  # Número de características
prob_nan = 0.1  # Probabilidad de que una respuesta sea no contestada

# Categorías de preguntas
categorias = [
    "Diseño", "Rendimiento", "Comodidad", "Seguridad", "Eficiencia", "Tecnología", "Precio y Mantenimiento"
]

# Generar nombres de preguntas dentro de cada categoría
preguntas = []
num_preguntas_por_categoria = num_features // len(categorias)

for i, categoria in enumerate(categorias):
    for j in range(num_preguntas_por_categoria):
        preguntas.append(f"{categoria} - Pregunta {j+1}")

# Si hay sobrantes, agregarlos al final
while len(preguntas) < num_features:
    preguntas.append(f"{categorias[len(preguntas) % len(categorias)]} - Pregunta Extra")

# Generar respuestas aleatorias entre 0 y 5
data = []
random.seed(42)  # Para reproducibilidad

for i in range(1, num_respuestas + 1):
    fila = [str(i)]  # ID único
    for _ in range(num_features):
        if random.random() < prob_nan:
            fila.append('')  # Representando NaN como cadena vacía
        else:
            fila.append(str(random.randint(0, 5)))
    data.append(fila)

# Guardar en un archivo CSV
with open("encuesta_autos.csv", "w", encoding="utf-8") as file:
    file.write("ID," + ",".join(preguntas) + "\n")
    for fila in data:
        file.write(",".join(fila) + "\n")

print("Archivo 'encuesta_autos.csv' generado correctamente.")

# Mostrar las primeras filas
for fila in data[:5]:
    print(fila)
