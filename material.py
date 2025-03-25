import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_tuner import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from scipy.optimize import differential_evolution, dual_annealing
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset for training.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: Scaled feature matrix (X), target variable (y), and preprocessor object.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)

        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least two columns (features and target).")

        # Handle missing values (if any)
        df = df.dropna()

        # Separate features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].values

        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Fit and transform the data
        X_preprocessed = preprocessor.fit_transform(X)

        # Outlier detection and removal (using Z-score)
        from scipy.stats import zscore
        z_scores = np.abs(zscore(X_preprocessed))
        X_preprocessed = X_preprocessed[(z_scores < 3).all(axis=1)]

        return X_preprocessed, y, preprocessor
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}")
        logging.error(f"Error loading or preprocessing data: {e}")
        return None, None, None

## Build a customizable neural network model
def build_model(hp=None, input_shape=None):
    """
    Build and compile a neural network model for regression.

    Args:
        hp (keras_tuner.HyperParameters): Optional hyperparameter tuner object.
        input_shape (int): Number of input features.

    Returns:
        keras.Sequential: Compiled Keras model.
    """
    if hp:
        # Use hyperparameter tuning
        units_1 = hp.Int('units_1', min_value=64, max_value=256, step=32)
        units_2 = hp.Int('units_2', min_value=32, max_value=128, step=16)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    else:
        # Default values
        units_1, units_2, dropout_rate, learning_rate = 128, 64, 0.2, 1e-3

    model = Sequential([
        Dense(units_1, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units_2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

## Hyperparameter tuning using Bayesian Optimization
def tune_hyperparameters(X_train, y_train, input_shape):
    """
    Perform hyperparameter tuning using Bayesian Optimization.

    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target values.
        input_shape (int): Number of input features.

    Returns:
        keras.Sequential: Best model after tuning.
    """
    def model_builder(hp):
        return build_model(hp, input_shape)

    tuner = BayesianOptimization(
        model_builder,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=2,
        directory='hyperparameter_tuning',
        project_name='concrete_mix_optimizer'
    )

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    st.info(f"Best hyperparameters: Units layer 1: {best_hps.get('units_1')}, Units layer 2: {best_hps.get('units_2')}, Dropout rate: {best_hps.get('dropout_rate')}, Learning rate: {best_hps.get('learning_rate')}")
    return tuner.get_best_models(num_models=1)[0]

## Train the model
def train_model(X, y, epochs=100, batch_size=16):
    """
    Train the neural network model on the given dataset.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target values.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        keras.Sequential: Trained Keras model.
        dict: Training history for visualization.
    """
    model = build_model(input_shape=X.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min')
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    return model, history.history

## Visualize training history
def plot_training_history(history):
    """
    Plot the training and validation loss/metrics over epochs.

    Args:
        history (dict): Training history.
    """
    st.subheader("Training History")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax[0].plot(history['loss'], label='Training Loss')
    ax[0].plot(history['val_loss'], label='Validation Loss')
    ax[0].set_title('Loss Over Epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # MAE plot
    ax[1].plot(history['mae'], label='Training MAE')
    ax[1].plot(history['val_mae'], label='Validation MAE')
    ax[1].set_title('Mean Absolute Error Over Epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('MAE')
    ax[1].legend()

    st.pyplot(fig)

## Objective function for genetic algorithm
def objective_function(mix):
    """
    Objective function for genetic algorithm optimization.

    Args:
        mix (list): A candidate concrete mix design.

    Returns:
        float: The absolute difference between target and predicted compressive strength.
    """
    mix = np.array(mix).reshape(1, -1)
    return abs(target_strength - model.predict(mix)[0][0])

## Genetic algorithm optimization
def optimize_mix():
    """
    Optimize the concrete mix design using a genetic algorithm.

    Returns:
        numpy.ndarray: Optimal mix design.
    """
    bounds = [(250, 500), (140, 220), (600, 1200), (0, 10)]  # Cement, Water, Aggregate, Admixture
    result = differential_evolution(objective_function, bounds)
    return result.x

## Bayesian Optimization for hyperparameter tuning
def bayesian_optimization(X_train, y_train, input_shape):
    """
    Perform hyperparameter tuning using Bayesian Optimization.

    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target values.
        input_shape (int): Number of input features.

    Returns:
        keras.Sequential: Best model after tuning.
    """
    def model_builder(hp):
        model = Sequential([
            Dense(hp.Int('units_1', min_value=64, max_value=256, step=32), activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)),
            Dense(hp.Int('units_2', min_value=32, max_value=128, step=16), activation='relu'),
            BatchNormalization(),
            Dropout(hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), loss='mse', metrics=['mae'])
        return model

    tuner = BayesianOptimization(
        model_builder,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=2,
        directory='hyperparameter_tuning',
        project_name='concrete_mix_optimizer_bayesian'
    )

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    st.info(f"Best hyperparameters: Units layer 1: {best_hps.get('units_1')}, Units layer 2: {best_hps.get('units_2')}, Dropout rate 1: {best_hps.get('dropout_rate_1')}, Dropout rate 2: {best_hps.get('dropout_rate_2')}, Learning rate: {best_hps.get('learning_rate')}")
    return tuner.get_best_models(num_models=1)[0]

## Streamlit Dashboard
st.set_page_config(page_title="AI-Based Concrete Mix Optimizer", layout="wide", page_icon="ahsankarya.ico")

# Menampilkan logo Ahsan Karya di sidebar
st.sidebar.image("ahsantech.png", use_container_width=True)
st.sidebar.markdown("---")  # Garis pemisah

# Streamlit Page Configuration
st.sidebar.title("🤖 AI-Based Concrete Mix Optimizer")
st.sidebar.write("### Information")

# Pilihan bahasa
language = st.sidebar.radio("🌍 Select Language / Pilih Bahasa", ("English", "Bahasa Indonesia"))

# Deskripsi berdasarkan bahasa yang dipilih
if language == "English":
    description = """
    <div style="text-align: justify;">
    ✨ <b>AI-Based Concrete Mix Optimizer</b> is an advanced <i>machine learning</i>-powered application designed to analyze, predict, and optimize concrete mix compositions to achieve <b>maximum compressive strength</b>. By considering key factors such as <b>cement, water, aggregate, and admixture</b>, this application assists engineers and researchers in creating concrete that is <b>stronger, more efficient, and cost-effective</b>.  

    🧠 Built with <i>Python</i> and powered by <i>Neural Networks (Deep Learning)</i> and <i>Genetic Algorithms</i>, this application has been trained on extensive construction material datasets to deliver <b>fast, accurate, and data-driven predictions</b>. Combining cutting-edge AI with interactive visualizations, <b>AI-Based Concrete Mix Optimizer</b> brings innovation to civil engineering and material science, enabling the design of concrete that is <b>more durable, economical, and environmentally friendly</b>. 🚀🏗️  
    </div>
    """
else:
    description = """
    <div style="text-align: justify;">
    ✨ <b>AI-Based Concrete Mix Optimizer</b> adalah aplikasi canggih berbasis <i>machine learning</i> yang dirancang untuk menganalisis, memprediksi, dan mengoptimalkan komposisi campuran beton guna mencapai <b>kekuatan tekan maksimal</b>. Dengan mempertimbangkan faktor-faktor seperti <b>semen, air, agregat, dan admixture</b>, aplikasi ini membantu insinyur dan peneliti menciptakan beton yang lebih <b>kuat, efisien, dan hemat biaya</b>.  

    🧠 Dibangun dengan <i>Python</i> dan didukung oleh <i>Neural Network (Deep Learning)</i> serta <i>Genetic Algorithm</i>, aplikasi ini telah dilatih menggunakan dataset material konstruksi untuk memberikan prediksi yang <b>cepat, akurat, dan berbasis data nyata</b>. Dengan kombinasi AI mutakhir dan visualisasi interaktif, <b>AI-Based Concrete Mix Optimizer</b> membawa inovasi baru dalam dunia teknik sipil dan material konstruksi, memungkinkan desain beton yang lebih <b>tahan lama, ekonomis, dan ramah lingkungan</b>. 🚀🏗️  
    </div>
    """

st.sidebar.markdown(description, unsafe_allow_html=True)

# Pemisah
st.sidebar.markdown("----")

# Informasi Pengembang
st.sidebar.write("### 👨‍💻 Developer Information")
st.sidebar.write("**Name:** Fathan Naufal Ahsan")
st.sidebar.write("**Brand:** Ahsan Karya")
st.sidebar.write("**Email:** [fathannaufalahsan.18@gmail.com](mailto:fathannaufalahsan.18@gmail.com)")

# Pemisah
st.sidebar.markdown("----")

# Pilihan Bahasa dengan Unique Key
language = st.sidebar.radio("🌍 Select Language / Pilih Bahasa", ("English", "Bahasa Indonesia"), key="language_selector")

# Noted Section berdasarkan bahasa yang dipilih
if language == "English":
    note_text = """
    **📝 Important Notes:**  
    1️⃣ If a **server error** occurs, please **refresh the page**. 🔄  
    2️⃣ The process may take **1-2 minutes** as the system processes **300 data samples**. Please be patient. ⏳  
    """
else:
    note_text = """
    **📝 Catatan Penting:**  
    1️⃣ Jika terjadi **error pada server**, silakan **muat ulang halaman**. 🔄  
    2️⃣ Proses memerlukan waktu **1-2 menit** karena sistem memproses **300 sampel data**. Harap bersabar. ⏳  
    """
# Menampilkan Noted Section
st.sidebar.markdown(note_text)

# Pemisah
st.sidebar.markdown("----")

# Sidebar for input parameters
with st.sidebar:
    st.header("🔧 Input Parameters")
    target_strength = st.slider("Target Compressive Strength (MPa)", 30.0, 80.0, 25.0)

    # Pilihan sumber dataset
    dataset_option = st.radio("Select Dataset Source", ("Use Default Dataset", "Upload Your Own CSV"))

    # Inisialisasi uploaded_file
    uploaded_file = None

    if dataset_option == "Upload Your Own CSV":
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    # Tentukan jalur dataset yang digunakan
    dataset_path = uploaded_file if uploaded_file is not None else "dataset.csv"

    epochs = st.slider("Number of Training Epochs", 10, 500, 100)
    batch_size = st.slider("Batch Size", 8, 64, 16)
    optimize_btn = st.button("🚀 Optimize Mix")

# Custom CSS untuk tampilan modern dan canggih
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

        /* Warna latar belakang */
        body {
            background-color: #0d1117;
            color: #ffffff;
        }

        /* Animasi Fade-in */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Animasi Glow */
        @keyframes glow {
            0% { text-shadow: 0 0 5px #33ccff, 0 0 10px #33ccff, 0 0 15px #33ccff; }
            50% { text-shadow: 0 0 10px #00e6e6, 0 0 20px #00e6e6, 0 0 30px #00e6e6; }
            100% { text-shadow: 0 0 5px #33ccff, 0 0 10px #33ccff, 0 0 15px #33ccff; }
        }

        /* Judul */
        .title {
            text-align: center;
            color: #33ccff;
            font-size: 3rem;
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            animation: fadeIn 2s ease-in-out, glow 3s infinite alternate;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #00e6e6;
            font-size: 1.2rem;
            font-family: 'Poppins', sans-serif;
            font-weight: 400;
            animation: fadeIn 3s ease-in-out;
        }

        /* Garis pemisah */
        .separator {
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, #33ccff, #00e6e6);
            margin: 20px auto;
            animation: fadeIn 1.5s ease-in-out;
        }

        /* Tombol interaktif */
        .ai-button {
            display: block;
            width: 250px;
            margin: 30px auto;
            padding: 15px;
            text-align: center;
            font-size: 1rem;
            font-family: 'Poppins', sans-serif;
            color: #fff;
            background: linear-gradient(90deg, #33ccff, #00e6e6);
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: 0.3s ease-in-out;
            box-shadow: 0 0 10px #00e6e6;
        }

        .ai-button:hover {
            background: linear-gradient(90deg, #00e6e6, #33ccff);
            box-shadow: 0 0 20px #00e6e6;
        }
    </style>

    <h1 class="title">🤖 AI-Based Concrete Mix Optimizer</h1>
    <h4 class="subtitle">Optimize your concrete mix design with AI-powered technology</h4>
    <div class="separator"></div>
    """,
    unsafe_allow_html=True
)

# ✅ Tambahkan Status Loading yang Lebih Profesional
if dataset_path and optimize_btn:
    if dataset_option == "Use Default Dataset" and not os.path.exists(dataset_path):
        st.error(f"❌ Default dataset `{dataset_path}` not found. Please upload a dataset instead.")
    else:
        st.info("📂 **Loading and preprocessing dataset...**")
        X, y, preprocessor = load_and_preprocess_data(dataset_path)

        if X is not None and y is not None:
            with st.spinner("🧠 **Training AI model, please wait...**"):
                model, history = train_model(X, y, epochs=epochs, batch_size=batch_size)

            # ✅ Bungkus Visualisasi dalam Container untuk Tampilan Lebih Rapi
            with st.container():
                st.success("✅ **Training Completed!**")
                plot_training_history(history)

            st.info("🔍 **Optimizing concrete mix...**")
            optimal_mix = optimize_mix()

            # ✅ Hasil Optimasi dalam Kotak Tersendiri
            with st.container():
                st.success("🎉 **Optimization Complete!**")
                st.subheader("🏗️ Recommended Mix Composition")
                st.write(f"🧪 **Cement**: {optimal_mix[0]:.2f} kg")
                st.write(f"💧 **Water**: {optimal_mix[1]:.2f} kg")
                st.write(f"🪨 **Aggregate**: {optimal_mix[2]:.2f} kg")
                st.write(f"🧂 **Admixture**: {optimal_mix[3]:.2f} kg")

                # ✅ Animasi Perayaan
                st.balloons()

            # ✅ Visualisasi dalam Container Terpisah
            with st.container():
                st.subheader("📊 **Visual Representation of Optimal Mix**")
                mix_labels = ['Cement', 'Water', 'Aggregate', 'Admixture']
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=mix_labels, y=optimal_mix, palette="coolwarm", ax=ax)
                ax.set_title("Optimal Concrete Mix Composition", fontsize=14, color='#333')
                ax.set_ylabel("Amount (kg)", fontsize=12, color='#555')
                st.pyplot(fig)

            # ✅ Evaluasi Model
            with st.container():
                st.subheader("📊 **Model Evaluation Metrics**")
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                st.write(f"Mean Squared Error (MSE): {mse:.4f}")
                st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
                st.write(f"R-squared (R²): {r2:.4f}")

            # ✅ Visualisasi 3D
            with st.container():
                st.subheader("📊 **3D Visualization of Predicted vs Actual Values**")
                fig = px.scatter_3d(
                    x=y, y=y_pred.flatten(), z=np.arange(len(y)),
                    labels={'x': 'Actual', 'y': 'Predicted', 'z': 'Sample Index'},
                    title="3D Scatter Plot of Actual vs Predicted Values"
                )
                st.plotly_chart(fig)

        else:
            st.error("⚠️ **Failed to load or preprocess the dataset.** Please check the uploaded file.")

else:
    if not dataset_path:
        st.warning("⚠️ **Please upload a dataset to proceed.**")
    if not optimize_btn:
        st.info("🔧 **Adjust the parameters and click 'Optimize Mix' to start.**")
