import fitz  # PyMuPDF para leer PDF
import re    # Expresiones regulares para limpiar texto
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

class PredictorTexto:
    def __init__(self, suavizado=1.0):
        self.modelos = {}  # Almacenar múltiples modelos (bigramas, trigramas, etc.)
        self.vocabulario = set()
        self.suavizado = suavizado  # Suavizado de Laplace
        self.tokens = []
        self.probabilidades_priori = {}  # Para implementación bayesiana
        self.freq_palabras = Counter()  # Frecuencias de palabras individuales
        
    def leer_pdfs(self, lista_rutas):
        #///Lee múltiples archivos PDF y extrae el texto///#
        texto_completo = ""
        archivos_leidos = 0
        
        for ruta in lista_rutas:
            try:
                doc = fitz.open(ruta)
                for pagina in doc:
                    texto_completo += pagina.get_text()
                doc.close()
                archivos_leidos += 1
                print(f"✓ Archivo leído: {ruta}")
            except Exception as e:
                print(f"✗ Error al leer {ruta}: {e}")
        
        print(f"Total de archivos procesados: {archivos_leidos}/{len(lista_rutas)}")
        return texto_completo
    
    def limpiar_tokenizar(self, texto):
        #///Limpia y tokeniza el texto de manera más robusta///#
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Conservar algunos signos de puntuación importantes
        texto = re.sub(r'[^\w\s\.\!\?\;]', ' ', texto)
        
        # Reemplazar múltiples espacios por uno solo
        texto = re.sub(r'\s+', ' ', texto)
        
        # Dividir en tokens
        tokens = texto.split()
        
        # Filtrar tokens válidos
        tokens = [token for token in tokens if len(token) > 1 and re.match(r'^[a-záéíóúüñ]+', token)]
        
        return tokens
    
    def construir_modelos_multiples(self, tokens, n_max=4):
        #///Construye múltiples modelos de n-gramas y calcula probabilidades priori///#
        self.tokens = tokens
        self.vocabulario = set(tokens)
        self.freq_palabras = Counter(tokens)
        
        # Calcular probabilidades priori (frecuencia de cada palabra)
        total_palabras = len(tokens)
        self.probabilidades_priori = {
            palabra: freq / total_palabras 
            for palabra, freq in self.freq_palabras.items()
        }
        
        print(f"Construyendo modelos de n-gramas (2 hasta {n_max})...")
        
        for n in range(2, n_max + 1):
            print(f"  - Construyendo modelo {n}-grama...")
            self.modelos[n] = self._construir_modelo_ngrama(tokens, n)
        
        print(f"Vocabulario total: {len(self.vocabulario)} palabras únicas")
    
    def _construir_modelo_ngrama(self, tokens, n):
        #///Construye un modelo de n-gramas con suavizado de Laplace//#
        if len(tokens) < n:
            return defaultdict(lambda: defaultdict(float))
            
        ngramas = [(tuple(tokens[i:i+n-1]), tokens[i+n-1]) 
                   for i in range(len(tokens)-n+1)]
        
        conteo_ngramas = Counter(ngramas)
        conteo_prefijos = Counter([ng[0] for ng in ngramas])
        
        modelo = defaultdict(lambda: defaultdict(float))
        
        # Aplicar suavizado de Laplace
        for (prefijo, siguiente), count in conteo_ngramas.items():
            probabilidad = (count + self.suavizado) / (conteo_prefijos[prefijo] + self.suavizado * len(self.vocabulario))
            modelo[prefijo][siguiente] = probabilidad
        
        return modelo
    
    def calcular_probabilidad_bayesiana(self, palabra, contexto, n):
        #//Teorema de Bayes para calcular P(contexto|palabra)  P(contexto|palabra) = P(palabra|contexto) * P(contexto) / P(palabra) ///#
        if n not in self.modelos or contexto not in self.modelos[n]:
            return 0.0
        
        # P(palabra|contexto) - del modelo de n-gramas
        p_palabra_dado_contexto = self.modelos[n][contexto].get(palabra, 0.0)
        
        # P(contexto) - frecuencia del contexto
        contexto_str = ' '.join(contexto)
        p_contexto = self._calcular_prob_contexto(contexto_str)
        
        # P(palabra) - probabilidad priori
        p_palabra = self.probabilidades_priori.get(palabra, 1e-10)
        
        # Aplicar Teorema de Bayes
        if p_palabra > 0:
            p_contexto_dado_palabra = (p_palabra_dado_contexto * p_contexto) / p_palabra
        else:
            p_contexto_dado_palabra = 0.0
        
        return p_contexto_dado_palabra
    
    def _calcular_prob_contexto(self, contexto_str):
        #/// Calcula la probabilidad de un contexto específico ///#
        # Buscar cuántas veces aparece este contexto en el texto
        contexto_count = 0
        tokens_texto = ' '.join(self.tokens)
        
        # Método simplificado: contar ocurrencias del contexto
        if contexto_str in tokens_texto:
            contexto_count = tokens_texto.count(contexto_str)
        
        # Probabilidad = count / total_posibles_contextos
        total_contextos = max(1, len(self.tokens) - len(contexto_str.split()) + 1)
        return (contexto_count + self.suavizado) / (total_contextos + self.suavizado)
    
    def predecir_con_bayes(self, texto_entrada, num_palabras=5, usar_bayes=True):
        #///Predice texto usando el enfoque bayesiano//#
        texto_actual = texto_entrada.lower().strip()
        historial = texto_actual.split()
        
        n_optimo = self.calcular_n_optimo(texto_entrada)
        print(f"Usando modelo {n_optimo}-grama {'con Bayes' if usar_bayes else 'tradicional'}")
        
        modelo = self.modelos[n_optimo]
        
        for iteracion in range(num_palabras):
            if len(historial) < n_optimo - 1:
                print(f"Contexto insuficiente en iteración {iteracion + 1}")
                break
            
            contexto = tuple(historial[-(n_optimo - 1):])
            
            if contexto in modelo and modelo[contexto]:
                if usar_bayes:
                    # Usar enfoque bayesiano
                    siguiente_palabra = self._seleccion_bayesiana(contexto, modelo[contexto], n_optimo)
                else:
                    # Usar enfoque tradicional
                    siguiente_palabra = max(modelo[contexto], key=modelo[contexto].get)
                
                if siguiente_palabra:
                    texto_actual += " " + siguiente_palabra
                    historial.append(siguiente_palabra)
                else:
                    break
            else:
                siguiente_palabra = self._estrategia_respaldo(historial)
                if siguiente_palabra:
                    texto_actual += " " + siguiente_palabra
                    historial.append(siguiente_palabra)
                else:
                    break
        
        return texto_actual
    
    def _seleccion_bayesiana(self, contexto, opciones, n):
        #///Selecciona la palabra usando probabilidades bayesianas///#
        scores_bayesianos = {}
        
        for palabra in opciones:
            # Combinar probabilidad del modelo con probabilidad bayesiana
            prob_ngram = opciones[palabra]
            prob_bayes = self.calcular_probabilidad_bayesiana(palabra, contexto, n)
            
            # Score combinado (puedes ajustar los pesos)
            score_combinado = 0.7 * prob_ngram + 0.3 * prob_bayes
            scores_bayesianos[palabra] = score_combinado
        
        if scores_bayesianos:
            return max(scores_bayesianos, key=scores_bayesianos.get)
        return None
    
    def calcular_n_optimo(self, texto_entrada):
        #///Calcula el valor óptimo de n basado en la longitud de la entrada///#
        palabras_entrada = len(texto_entrada.strip().split())
        
        if palabras_entrada == 1:
            n_optimo = 2
        elif palabras_entrada <= 3:
            n_optimo = palabras_entrada + 1
        else:
            n_optimo = min(4, palabras_entrada + 1)
        
        if n_optimo not in self.modelos:
            n_optimo = max([n for n in self.modelos.keys() if n <= n_optimo], default=2)
        
        return n_optimo
    
    def _seleccion_probabilistica(self, opciones):
        #///Selecciona una palabra basada en su probabilidad//#
        palabras = list(opciones.keys())
        probabilidades = list(opciones.values())
        
        suma_prob = sum(probabilidades)
        if suma_prob > 0:
            probabilidades = [p/suma_prob for p in probabilidades]
            return np.random.choice(palabras, p=probabilidades)
        return palabras[0] if palabras else None
    
    def _estrategia_respaldo(self, historial):
        #///Estrategia de respaldo cuando no se encuentra el contexto//#
        for n_respaldo in sorted([n for n in self.modelos.keys()], reverse=True):
            if len(historial) >= n_respaldo - 1:
                contexto = tuple(historial[-(n_respaldo - 1):])
                if contexto in self.modelos[n_respaldo] and self.modelos[n_respaldo][contexto]:
                    opciones = self.modelos[n_respaldo][contexto]
                    return max(opciones, key=opciones.get)
        
        if self.tokens:
            return self.freq_palabras.most_common(1)[0][0]
        
        return None
    
    def obtener_estadisticas(self):
        #///Obtiene estadísticas del modelo///#
        stats = {
            'vocabulario_size': len(self.vocabulario),
            'total_tokens': len(self.tokens),
            'modelos_construidos': list(self.modelos.keys()),
            'palabras_mas_frecuentes': self.freq_palabras.most_common(10)
        }
        
        for n in self.modelos.keys():
            contextos = len(self.modelos[n])
            stats[f'contextos_{n}grama'] = contextos
        
        return stats
    
    def comparar_enfoques(self, texto_entrada, num_palabras=5):
        #///Compara los resultados entre el enfoque tradicional y bayesiano//#
        print(f"\n=== COMPARACIÓN DE ENFOQUES ===")
        print(f"Entrada: '{texto_entrada}'")
        
        # Enfoque tradicional
        resultado_tradicional = self.predecir_con_bayes(
            texto_entrada, num_palabras, usar_bayes=False
        )
        print(f"Tradicional: {resultado_tradicional}")
        
        # Enfoque bayesiano
        resultado_bayesiano = self.predecir_con_bayes(
            texto_entrada, num_palabras, usar_bayes=True
        )
        print(f"Bayesiano:   {resultado_bayesiano}")
        
        return resultado_tradicional, resultado_bayesiano

# Ejemplo de uso mejorado
def main():
    # Crear el predictor
    predictor = PredictorTexto(suavizado=1.0)

    rutas = [
        r'PDF\25-preguntas-tipicas-entrevista.pdf',
        r'PDF\Cien_años_de_soledad.pdf',
        r'PDF\coonversacion.pdf',
        r'PDF\quijote.pdf'
    ]
    print("=== CARGANDO Y PROCESANDO PDFs ===")
    texto = predictor.leer_pdfs(rutas)
    tokens = predictor.limpiar_tokenizar(texto)

    
    # Construir modelos
    print("\n=== CONSTRUYENDO MODELOS ===")
    predictor.construir_modelos_multiples(tokens, n_max=4)
    
    # Mostrar estadísticas
    print("\n=== ESTADÍSTICAS DEL MODELO ===")
    stats = predictor.obtener_estadisticas()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Comparar enfoques
    entradas_prueba = [
        "facer"
    ]
    
    for entrada in entradas_prueba:
        predictor.comparar_enfoques(entrada, num_palabras=3)

if __name__ == "__main__":
    main()