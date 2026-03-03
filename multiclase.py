import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import warnings
warnings.filterwarnings('ignore')

class PenguinClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Pingüinos - One-vs-Rest")
        self.root.geometry("1200x700")
        
        # Variables de la aplicación
        self.df = None
        self.df_clean = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.species_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
        
        # Configurar estilo
        self.setup_styles()
        
        # Crear interfaz
        self.create_widgets()
        
        # Cargar datos automáticamente
        self.load_initial_data()
    
    def setup_styles(self):
        """Configurar estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Success.TButton', font=('Arial', 10))
    
    def create_widgets(self):
        """Crear los widgets de la interfaz"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        title_label = ttk.Label(main_frame, text="Clasificador de Especies de Pingüinos", 
                               style='Title.TLabel')
        title_label.pack(pady=10)
        
        # Subtítulo con explicación OvR
        subtitle = ttk.Label(main_frame, 
                            text="One-vs-Rest: La probabilidad más alta gana",
                            font=('Arial', 10, 'italic'))
        subtitle.pack()
        
        # Notebook (pestañas)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Crear pestañas
        self.create_data_tab()
        self.create_analysis_tab()
        self.create_model_tab()
        self.create_prediction_tab()
        self.create_ovr_explanation_tab()
        
        # Barra de estado
        self.status_bar = ttk.Label(main_frame, text="Listo", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, pady=5)
    
    def create_data_tab(self):
        """Pestaña de datos y limpieza"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📊 Datos y Limpieza")
        
        # Frame para botones
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Cargar CSV", 
                  command=self.load_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Limpiar Datos", 
                  command=self.clean_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Ver Estadísticas", 
                  command=self.show_statistics).pack(side=tk.LEFT, padx=5)
        
        # Frame para mostrar datos
        data_frame = ttk.LabelFrame(tab, text="Datos Cargados", padding="5")
        data_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Treeview para mostrar datos
        columns = ('CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass', 'Species')
        self.data_tree = ttk.Treeview(data_frame, columns=columns, show='headings', height=15)
        
        # Configurar columnas
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=120)
        
        # Scrollbars
        vsb = ttk.Scrollbar(data_frame, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(data_frame, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        data_frame.grid_rowconfigure(0, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)
        
        # Info de valores nulos
        self.null_info = ttk.Label(tab, text="", foreground="red")
        self.null_info.pack(pady=5)
    
    def create_analysis_tab(self):
        """Pestaña de análisis exploratorio"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📈 Análisis Exploratorio")
        
        # Frame para botones de gráficos
        button_frame = ttk.Frame(tab)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="Distribución por Especie", 
                  command=self.plot_species_distribution).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Boxplots", 
                  command=self.plot_boxplots).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Matriz de Correlación", 
                  command=self.plot_correlation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Pairplot", 
                  command=self.plot_pairplot).pack(side=tk.LEFT, padx=5)
        
        # Frame para el gráfico
        self.plot_frame = ttk.LabelFrame(tab, text="Visualizaciones", padding="5")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Figura de matplotlib
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_model_tab(self):
        """Pestaña de entrenamiento del modelo"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🤖 Modelo OvR")
        
        # Frame de controles
        control_frame = ttk.LabelFrame(tab, text="Configuración", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Tamaño de prueba (%):").grid(row=0, column=0, padx=5)
        self.test_size = ttk.Combobox(control_frame, values=[20, 25, 30, 35, 40], width=10)
        self.test_size.set(30)
        self.test_size.grid(row=0, column=1, padx=5)
        
        ttk.Button(control_frame, text="Entrenar Modelo", 
                  command=self.train_model).grid(row=0, column=2, padx=20)
        
        # Frame para resultados
        results_frame = ttk.LabelFrame(tab, text="Resultados del Modelo", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Texto para resultados
        self.results_text = ScrolledText(results_frame, height=15, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        # Matriz de confusión
        ttk.Button(tab, text="Mostrar Matriz de Confusión", 
                  command=self.show_confusion_matrix).pack(pady=5)
    
    def create_prediction_tab(self):
        """Pestaña de predicción"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔮 Predecir")
        
        # Frame para entrada de datos
        input_frame = ttk.LabelFrame(tab, text="Características del Pingüino", padding="15")
        input_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Crear campos de entrada
        self.inputs = {}
        features = [
            ('Culmen Length (mm):', 40.0),
            ('Culmen Depth (mm):', 17.0),
            ('Flipper Length (mm):', 195.0),
            ('Body Mass (g):', 4000.0)
        ]
        
        for i, (label, default) in enumerate(features):
            ttk.Label(input_frame, text=label).grid(row=i, column=0, sticky='w', pady=5)
            self.inputs[label] = ttk.Entry(input_frame, width=15)
            self.inputs[label].grid(row=i, column=1, padx=10, pady=5)
            self.inputs[label].insert(0, str(default))
        
        # Botón de predicción
        ttk.Button(input_frame, text="Predecir Especie", 
                  command=self.predict_species).grid(row=len(features), column=0, columnspan=2, pady=15)
        
        # Frame para resultados
        result_frame = ttk.LabelFrame(tab, text="Resultado de la Predicción", padding="15")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Labels para mostrar resultados
        self.prediction_label = ttk.Label(result_frame, text="", font=('Arial', 14, 'bold'))
        self.prediction_label.pack(pady=10)
        
        # Frame para barras de probabilidad
        self.prob_frame = ttk.Frame(result_frame)
        self.prob_frame.pack(fill=tk.X, pady=10)
        
        # Crear barras de probabilidad para cada especie
        self.prob_bars = {}
        self.prob_labels = {}
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (sp, color) in enumerate(zip(['Adelie', 'Chinstrap', 'Gentoo'], colors)):
            frame = ttk.Frame(self.prob_frame)
            frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(frame, text=f"{sp}:", width=15).pack(side=tk.LEFT)
            
            # Barra de progreso
            bar = ttk.Progressbar(frame, length=300, mode='determinate')
            bar.pack(side=tk.LEFT, padx=10)
            self.prob_bars[sp] = bar
            
            # Label para porcentaje
            label = ttk.Label(frame, text="0%", width=10)
            label.pack(side=tk.LEFT)
            self.prob_labels[sp] = label
    
    def create_ovr_explanation_tab(self):
        """Pestaña explicativa de One-vs-Rest"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📚 ¿Cómo funciona OvR?")
        
        # Texto explicativo
        text_widget = ScrolledText(tab, wrap=tk.WORD, font=('Arial', 11))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        explanation = """
        🔍 ONE-VS-REST (OvR) EXPLICADO
        
        ¿Qué es?
        ---------
        One-vs-Rest (también conocido como One-vs-All) es una estrategia para usar 
        clasificadores binarios en problemas multiclase.
        
        ¿Cómo funciona?
        ----------------
        Para clasificar pingüinos en 3 especies, OvR crea 3 clasificadores binarios:
        
        1. 🤓 Clasificador 1: ¿Es Adelie? (vs Chinstrap + Gentoo)
        2. 🧐 Clasificador 2: ¿Es Chinstrap? (vs Adelie + Gentoo)
        3. 😎 Clasificador 3: ¿Es Gentoo? (vs Adelie + Chinstrap)
        
        Durante el entrenamiento:
        -------------------------
        • Cada clasificador aprende a separar UNA especie del resto
        • Se usan los mismos datos pero con diferentes etiquetas binarias
        
        Durante la predicción:
        ----------------------
        • Los 3 clasificadores calculan la probabilidad para su especie
        • Se obtienen 3 probabilidades: P(Adelie), P(Chinstrap), P(Gentoo)
        • La especie con la probabilidad MÁS ALTA es la predicción final
        
        Ejemplo Visual:
        ---------------
        Para un pingüino desconocido:
        
        Clasificador Adelie:    85% de ser Adelie    → 👍 Probabilidad alta
        Clasificador Chinstrap: 30% de ser Chinstrap → 👎 Probabilidad baja
        Clasificador Gentoo:    45% de ser Gentoo    → 👎 Probabilidad media
        
        ✅ RESULTADO: Especie ADELIE (gana la probabilidad más alta)
        
        Ventajas:
        ---------
        ✓ Simple y fácil de entender
        ✓ Funciona con cualquier clasificador binario
        ✓ Las probabilidades son comparables directamente
        ✓ Eficiente computacionalmente
        
        En este programa:
        -----------------
        Usamos Regresión Logística como clasificador binario base.
        El modelo OvR entrena 3 regresiones logísticas diferentes.
        ¡Pruébalo en la pestaña "Predecir"!
        """
        
        text_widget.insert('1.0', explanation)
        text_widget.configure(state='disabled')
    
    def load_initial_data(self):
        """Cargar datos iniciales del archivo"""
        try:
            self.df = pd.read_csv('penguins.csv')
            self.update_status("Datos cargados exitosamente")
            self.display_data()
            self.check_nulls()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
    
    def load_csv(self):
        """Cargar archivo CSV manualmente"""
        filename = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            try:
                self.df = pd.read_csv(filename)
                self.update_status(f"Datos cargados desde: {filename}")
                self.display_data()
                self.check_nulls()
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")
    
    def display_data(self):
        """Mostrar datos en el treeview"""
        if self.df is not None:
            # Limpiar treeview
            for row in self.data_tree.get_children():
                self.data_tree.delete(row)
            
            # Mostrar primeras 50 filas
            for idx, row in self.df.head(50).iterrows():
                values = [row['CulmenLength'], row['CulmenDepth'], 
                         row['FlipperLength'], row['BodyMass'], row['Species']]
                self.data_tree.insert('', 'end', values=values)
    
    def check_nulls(self):
        """Verificar y mostrar valores nulos"""
        if self.df is not None:
            nulls = self.df.isnull().sum().sum()
            if nulls > 0:
                self.null_info.config(
                    text=f"⚠️ Se encontraron {nulls} valores nulos. Usa 'Limpiar Datos' para eliminarlos.",
                    foreground="orange"
                )
            else:
                self.null_info.config(text="✅ No hay valores nulos", foreground="green")
    
    def clean_data(self):
        """Limpiar datos (eliminar nulos)"""
        if self.df is not None:
            before = len(self.df)
            self.df_clean = self.df.dropna()
            after = len(self.df_clean)
            
            self.update_status(f"Datos limpiados: {before} → {after} filas ({before-after} eliminadas)")
            messagebox.showinfo("Limpieza Completa", 
                              f"Se eliminaron {before-after} filas con valores nulos.\n"
                              f"Dataset limpio: {after} filas")
            
            # Actualizar vista
            self.display_data()
            self.check_nulls()
    
    def show_statistics(self):
        """Mostrar estadísticas descriptivas"""
        if self.df_clean is not None:
            stats = self.df_clean.describe()
            
            # Crear ventana de estadísticas
            stats_window = tk.Toplevel(self.root)
            stats_window.title("Estadísticas Descriptivas")
            stats_window.geometry("600x400")
            
            text = ScrolledText(stats_window, font=('Courier', 10))
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            text.insert('1.0', str(stats.round(2)))
            text.configure(state='disabled')
    
    def plot_species_distribution(self):
        """Graficar distribución de especies"""
        if self.df_clean is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Contar especies
            species_counts = self.df_clean['Species'].map(self.species_map).value_counts()
            
            # Crear gráfico de barras
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = ax.bar(species_counts.index, species_counts.values, color=colors)
            ax.set_title('Distribución de Especies', fontsize=14, fontweight='bold')
            ax.set_xlabel('Especie')
            ax.set_ylabel('Cantidad')
            
            # Agregar valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            self.figure.tight_layout()
            self.canvas.draw()
    
    def plot_boxplots(self):
        """Graficar boxplots de características"""
        if self.df_clean is not None:
            self.figure.clear()
            
            df_plot = self.df_clean.copy()
            df_plot['Species'] = df_plot['Species'].map(self.species_map)
            
            features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
            
            for i, feature in enumerate(features, 1):
                ax = self.figure.add_subplot(2, 2, i)
                
                # Crear boxplot
                df_plot.boxplot(column=feature, by='Species', ax=ax)
                ax.set_title(f'{feature}')
                ax.set_xlabel('')
            
            self.figure.suptitle('Distribución de Características por Especie', fontsize=14, fontweight='bold')
            self.figure.tight_layout()
            self.canvas.draw()
    
    def plot_correlation(self):
        """Graficar matriz de correlación"""
        if self.df_clean is not None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            # Calcular correlación
            features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
            corr = self.df_clean[features].corr()
            
            # Crear heatmap
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Configurar ejes
            ax.set_xticks(range(len(features)))
            ax.set_yticks(range(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_yticklabels(features)
            
            # Agregar valores
            for i in range(len(features)):
                for j in range(len(features)):
                    text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                                 ha='center', va='center', fontweight='bold')
            
            self.figure.colorbar(im, ax=ax)
            ax.set_title('Matriz de Correlación', fontsize=14, fontweight='bold')
            self.figure.tight_layout()
            self.canvas.draw()
    
    def plot_pairplot(self):
        """Graficar pairplot simplificado"""
        if self.df_clean is not None:
            self.figure.clear()
            
            df_plot = self.df_clean.copy()
            df_plot['Species'] = df_plot['Species'].map(self.species_map)
            
            features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
            colors = {'Adelie': '#FF6B6B', 'Chinstrap': '#4ECDC4', 'Gentoo': '#45B7D1'}
            
            n = len(features)
            for i in range(n):
                for j in range(n):
                    ax = self.figure.add_subplot(n, n, i*n + j + 1)
                    
                    if i == j:
                        # Histograma en diagonal
                        for species in colors:
                            data = df_plot[df_plot['Species'] == species][features[i]]
                            ax.hist(data, alpha=0.5, color=colors[species], label=species)
                    else:
                        # Scatter plot
                        for species in colors:
                            data = df_plot[df_plot['Species'] == species]
                            ax.scatter(data[features[j]], data[features[i]], 
                                     alpha=0.5, color=colors[species], s=20)
                    
                    if i == n-1:
                        ax.set_xlabel(features[j][:10], rotation=45)
                    if j == 0:
                        ax.set_ylabel(features[i][:10])
            
            self.figure.suptitle('Relación entre Características', fontsize=14, fontweight='bold')
            self.figure.tight_layout()
            self.canvas.draw()
    
    def train_model(self):
        """Entrenar el modelo One-vs-Rest"""
        if self.df_clean is None:
            messagebox.showwarning("Atención", "Primero debes limpiar los datos")
            return
        
        try:
            # Preparar datos
            features = ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'BodyMass']
            X = self.df_clean[features]
            y = self.df_clean['Species']
            
            # Dividir datos
            test_size = float(self.test_size.get()) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Escalar
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Guardar datos
            self.X_train = X_train_scaled
            self.X_test = X_test_scaled
            self.y_train = y_train
            self.y_test = y_test
            
            # Entrenar modelo
            base_lr = LogisticRegression(max_iter=1000, random_state=42)
            self.model = OneVsRestClassifier(base_lr)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluar
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Mostrar resultados
            self.results_text.delete('1.0', tk.END)
            self.results_text.insert('1.0', "="*60 + "\n")
            self.results_text.insert('1.0', "RESULTADOS DEL MODELO ONE-VS-REST\n")
            self.results_text.insert('1.0', "="*60 + "\n\n")
            
            self.results_text.insert('1.0', f"Precisión del modelo: {accuracy:.4f}\n\n")
            
            self.results_text.insert('1.0', "Reporte de Clasificación:\n")
            self.results_text.insert('1.0', classification_report(y_test, y_pred, 
                                                                  target_names=list(self.species_map.values())))
            
            self.update_status(f"Modelo entrenado - Precisión: {accuracy:.4f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el entrenamiento: {e}")
    
    def show_confusion_matrix(self):
        """Mostrar matriz de confusión"""
        if self.model is None:
            messagebox.showwarning("Atención", "Primero debes entrenar el modelo")
            return
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Crear heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=list(self.species_map.values()),
                   yticklabels=list(self.species_map.values()))
        
        ax.set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
        # Cambiar a la pestaña de análisis
        self.notebook.select(1)
    
    def predict_species(self):
        """Predecir especie para nuevos datos"""
        if self.model is None:
            messagebox.showwarning("Atención", "Primero debes entrenar el modelo")
            return
        
        try:
            # Obtener valores de entrada
            features = [
                float(self.inputs['Culmen Length (mm):'].get()),
                float(self.inputs['Culmen Depth (mm):'].get()),
                float(self.inputs['Flipper Length (mm):'].get()),
                float(self.inputs['Body Mass (g):'].get())
            ]
            
            # Crear DataFrame
            new_data = pd.DataFrame([features], 
                                   columns=['CulmenLength', 'CulmenDepth', 
                                           'FlipperLength', 'BodyMass'])
            
            # Escalar
            new_data_scaled = self.scaler.transform(new_data)
            
            # Predecir
            probas = self.model.predict_proba(new_data_scaled)[0]
            pred = self.model.predict(new_data_scaled)[0]
            
            # Mostrar resultados
            species_name = self.species_map[pred]
            self.prediction_label.config(
                text=f"✨ Especie Predicha: {species_name} ✨",
                foreground="green"
            )
            
            # Actualizar barras de probabilidad
            for i, (sp, bar) in enumerate(self.prob_bars.items()):
                prob_percent = probas[i] * 100
                bar['value'] = prob_percent
                self.prob_labels[sp].config(text=f"{prob_percent:.1f}%")
            
            # Explicación de OvR
            messagebox.showinfo("One-vs-Rest", 
                              f"Los 3 clasificadores votaron:\n\n"
                              f"• Adelie: {probas[0]:.3f}\n"
                              f"• Chinstrap: {probas[1]:.3f}\n"
                              f"• Gentoo: {probas[2]:.3f}\n\n"
                              f"👉 Gana: {species_name} (probabilidad más alta)")
            
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa valores numéricos válidos")
    
    def update_status(self, message):
        """Actualizar barra de estado"""
        self.status_bar.config(text=f"📌 {message}")

def main():
    root = tk.Tk()
    app = PenguinClassifierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()