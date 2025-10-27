import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

def leer_txt(file_path):
    try:
        with open(file_path, 'r') as file:
            filas = file.readlines()
    except FileNotFoundError:
        print("Error: No se pudo abrir el archivo")
        return None, None
    
    datos = [list(map(float, linea.strip().split())) for linea in filas]
    columnas = list(zip(*datos))
    return columnas

# Leer datos desde el archivo
archivo_txt = r"Grupo12.txt"
datos = leer_txt(archivo_txt)

if datos:
    x1 = datos[0]  # Primera columna
    x2 = datos[1]  # Segunda columna
    y = datos[2]  # Tercera columna
    
    # Normalizar datos
    def normalizar(lista):
        min_val, max_val = min(lista), max(lista)
        return [(xi - min_val) / (max_val - min_val) for xi in lista]
    
    x1_norm = normalizar(x1)
    x2_norm = normalizar(x2)
    y_norm = normalizar(y)

    class RegresionLineal:
        def _init_(self, x1, x2, y, alpha=0.001, delta=1e-6):
            self.x1 = x1
            self.x2 = x2
            self.y = y
            self.alpha = alpha
            self.delta = delta
            self.w0 = random.uniform(-1, 1)
            self.w1 = random.uniform(-1, 1)
            self.w2 = random.uniform(-1, 1)
        
        def hipotesis(self, i):
            return self.w0 + self.w1 * self.x1[i] + self.w2 * self.x2[i]
        
        def calcular_error(self):
            m = len(self.y)
            return sum((self.hipotesis(i) - self.y[i])**2 for i in range(m)) / (2 * m)
        
        def entrenar(self):
            error_anterior = float('inf')
            while True:
                m = len(self.y)
                grad_w0 = sum(self.hipotesis(i) - self.y[i] for i in range(m)) / m
                grad_w1 = sum((self.hipotesis(i) - self.y[i]) * self.x1[i] for i in range(m)) / m
                grad_w2 = sum((self.hipotesis(i) - self.y[i]) * self.x2[i] for i in range(m)) / m
                
                self.w0 -= self.alpha * grad_w0
                self.w1 -= self.alpha * grad_w1
                self.w2 -= self.alpha * grad_w2
                
                error_actual = self.calcular_error()
                if abs(error_anterior - error_actual) < self.delta:
                    break
                error_anterior = error_actual
        
        def obtener_pesos(self):
            return self.w0, self.w1, self.w2

    regresor = RegresionLineal(x1_norm, x2_norm, y_norm, alpha=0.1, delta=1e-6)
    regresor.entrenar()
    w0, w1, w2 = regresor.obtener_pesos()
    print(f"w0: {w0}, w1: {w1}, w2: {w2}")

    # Graficar resultados en 3D
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(x1_norm, x2_norm, y_norm, color='blue')

    # Generar puntos para la superficie
    x1_vals = []
    x2_vals = []
    y_vals = []

    x1_grid = [i / 10 for i in range(11)]
    x2_grid = [i / 10 for i in range(11)]

    for x1_val in x1_grid:
        for x2_val in x2_grid:
            x1_vals.append(x1_val)
            x2_vals.append(x2_val)
            y_vals.append(w0 + w1 * x1_val + w2 * x2_val)

    # Graficar superficie de regresión correctamente
    ax.plot_trisurf(x1_vals, x2_vals, y_vals, color='red', alpha=0.5)

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y')
    ax.set_title('Regresión Lineal 3D')

    # Gráfica X1 vs Y con línea de regresión
    ax2 = fig.add_subplot(132)
    ax2.scatter(x1_norm, y_norm, color='green', label='X1 vs Y')
    sorted_x1 = sorted(x1_norm)
    ax2.plot(sorted_x1, [w0 + w1 * xi for xi in sorted_x1], color='red', label='Regresión')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('Y')
    ax2.set_title('X1 vs Y')
    ax2.legend()
    
    # Gráfica X2 vs Y con línea de regresión
    ax3 = fig.add_subplot(133)
    ax3.scatter(x2_norm, y_norm, color='red', label='X2 vs Y')
    sorted_x2 = sorted(x2_norm)
    ax3.plot(sorted_x2, [w0 + w2 * xi for xi in sorted_x2], color='blue', label='Regresión')
    ax3.set_xlabel('X2')
    ax3.set_ylabel('Y')
    ax3.set_title('X2 vs Y')
    ax3.legend()
    
    plt.tight_layout()
plt.show()