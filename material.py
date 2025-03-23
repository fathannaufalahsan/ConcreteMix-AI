import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset for training.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        tuple: Scaled feature matrix (X), target variable (y), and scaler object.
    """
    try:
        # Load dataset
        df = pd.read_csv(file_path)

        if df.shape[1] < 2:
            raise ValueError("Dataset must have at least two columns (features and target).")

        # Handle missing values (if any)
        df = df.dropna()

        # Separate features and target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return X_scaled, y, scaler
    except Exception as e:
        st.error(f"Error loading or preprocessing data: {e}")
        return None, None, None

# Build a customizable neural network model
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
    else:
        # Default values
        units_1, units_2, dropout_rate = 128, 64, 0.2

    model = Sequential([
        Dense(units_1, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(units_2, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Hyperparameter tuning using keras-tuner
def tune_hyperparameters(X_train, y_train, input_shape):
    """
    Perform hyperparameter tuning using keras-tuner.

    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target values.
        input_shape (int): Number of input features.

    Returns:
        keras.Sequential: Best model after tuning.
    """
    def model_builder(hp):
        return build_model(hp, input_shape)

    tuner = RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='hyperparameter_tuning',
        project_name='concrete_mix_optimizer'
    )

    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    st.info(f"Best hyperparameters: Units layer 1: {best_hps.get('units_1')}, Units layer 2: {best_hps.get('units_2')}, Dropout rate: {best_hps.get('dropout_rate')}")
    return tuner.get_best_models(num_models=1)[0]

# Train the model
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
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    return model, history.history

# Visualize training history
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

# Objective function for genetic algorithm
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

# Genetic algorithm optimization
def optimize_mix():
    """
    Optimize the concrete mix design using a genetic algorithm.

    Returns:
        numpy.ndarray: Optimal mix design.
    """
    bounds = [(100, 500), (100, 250), (500, 1200), (0, 50)]  # Cement, Water, Aggregate, Admixture
    result = differential_evolution(objective_function, bounds)
    return result.x

# Streamlit Dashboard
st.set_page_config(page_title="AI-Based Concrete Mix Optimizer", layout="wide", page_icon="ahsankarya.ico")
st.title("🔬 AI-Based Concrete Mix Optimizer")
st.markdown("### Optimal concrete mix design using Deep Learning & Genetic Algorithms")


# Menampilkan logo Ahsan Karya di sidebar
st.sidebar.image("ahsankarya.png", use_container_width=True)
st.sidebar.markdown("---")  # Garis pemisah

# Streamlit Page Configuration
st.sidebar.title("🔬 AI-Based Concrete Mix Optimizer")
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
    2️⃣ The process may take **2 minutes** as the system processes **500 data samples**. Please be patient. ⏳  
    """
else:
    note_text = """
    **📝 Catatan Penting:**  
    1️⃣ Jika terjadi **error pada server**, silakan **muat ulang halaman**. 🔄  
    2️⃣ Proses memerlukan waktu **2 menit** karena sistem memproses **500 sampel data**. Harap bersabar. ⏳  
    """
# Menampilkan Noted Section
st.sidebar.markdown(note_text)

# Pemisah
st.sidebar.markdown("----")

# Sidebar for input parameters
with st.sidebar:
    st.header("🔧 Input Parameters")
    target_strength = st.slider("Target Compressive Strength (MPa)", 10.0, 100.0, 40.0)

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

# Main logic
if dataset_path and optimize_btn:
    # Cek apakah dataset bawaan tersedia jika user memilih "Use Default Dataset"
    if dataset_option == "Use Default Dataset" and not os.path.exists(dataset_path):
        st.error(f"❌ Default dataset `{dataset_path}` not found. Please upload a dataset instead.")
    else:
        # Load and preprocess data
        st.info("Loading and preprocessing dataset...")
        X, y, scaler = load_and_preprocess_data(dataset_path)

        if X is not None and y is not None:
            st.info("Training AI model, please wait...")
            with st.spinner("Training the model..."):
                model, history = train_model(X, y, epochs=epochs, batch_size=batch_size)

            # Visualize training history
            plot_training_history(history)

            # Optimize mix
            st.info("Optimizing concrete mix...")
            optimal_mix = optimize_mix()

            # Display results
            st.success("Optimization Complete!")
            st.subheader("Recommended Mix Composition:")
            st.write(f"🧪 **Cement**: {optimal_mix[0]:.2f} kg")
            st.write(f"💧 **Water**: {optimal_mix[1]:.2f} kg")
            st.write(f"🪨 **Aggregate**: {optimal_mix[2]:.2f} kg")
            st.write(f"🧂 **Admixture**: {optimal_mix[3]:.2f} kg")

            # Display balloons to celebrate completion
            st.balloons()

            # Add final visualization of the optimized mix
            st.subheader("Visual Representation of Optimal Mix")
            mix_labels = ['Cement', 'Water', 'Aggregate', 'Admixture']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x=mix_labels, y=optimal_mix, palette="viridis", ax=ax)
            ax.set_title("Optimal Concrete Mix Composition")
            ax.set_ylabel("Amount (kg)")
            st.pyplot(fig)

        else:
            st.error("Failed to load or preprocess the dataset. Please check the uploaded file.")

else:
    if not dataset_path:
        st.warning("Please upload a dataset to proceed.")
    if not optimize_btn:
        st.info("Adjust the parameters and click 'Optimize Mix' to start.")