# NEXUS-CNN: Futuristic AI Binary Image Classification Dashboard

A production-ready, highly interactive Streamlit application serving as a futuristic AI binary image classification dashboard (NEXUS-CNN). The design bypasses the default Streamlit appearance by injecting custom HTML and CSS to create a glassmorphic user interface complete with neon glow components, scanning animations, and visual metrics telemetry.

The CNN model architecture and historical training telemetry are directly derived from the deep learning training pipeline in `Binary_Image_Classification_using_CNN_Deep_Learning_Basics.ipynb`.

---

## 🌌 Key Features

1. **Futuristic Glassmorphic Design**: Customized CSS theme including glowing text, neon borders (`#00f3ff` cyan, `#ff007f` magenta), glassmorphism transparency, and smooth card hover animations.
2. **Interactive Neural Scanner**: A custom image uploader container displaying a simulated holographic scanning beam over uploaded images, generating sigmoid classification signals, confidence gauges, and latency reports.
3. **Telemetry & Feature Activation Maps**: Displays simulated high-contrast feature maps (Edge detection, Texture variance, Activation heatmaps) corresponding to intermediate Conv2D and Pooling layers.
4. **Historical & Live Training Analytics**:
   - High-fidelity interactive line charts plotting the 10-epoch training run metrics from the notebook.
   - An in-browser **Live Training Simulator** with parameters for learning rate, batch size, epochs, and optimizer to witness real-time backpropagation simulation.
5. **CNN Node Explorer**: A step-by-step interactive workflow mapping each layer of the deep CNN (Conv2D -> MaxPooling -> Flatten -> Dense -> Dense Output) with structural dimension details and param counts.
6. **Customizable Target Classes**: Adjust class boundary names dynamically through the Settings panel (default: `Female` and `Male`).
7. **Robust ML Engine Fallback**: Automatically compiles the Keras architecture from the notebook on startup if TensorFlow is loaded. If TensorFlow is absent, it seamlessly falls back to a custom matrix processor using `Pillow` and `NumPy` to guarantee execution safety.

---

## 🛠️ Local Installation

### Prerequisites
Make sure Python 3.8+ is installed on your local environment.

### Setup Steps
1. Navigate to the project directory:
   ```bash
   cd "Binary image classifier"
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If installing TensorFlow fails or is too slow, the requirements file handles optional configuration. The app will run in "Optimized Edge Simulation Core" mode.*

3. Start the dashboard server:
   ```bash
   streamlit run app.py
   ```

4. The dashboard will initialize and open automatically in your browser at `http://localhost:8501`.

---

## 🚀 Deployment Guidelines

### Streamlit Community Cloud
1. Push this directory structure to a GitHub repository:
   ```text
   ├── .streamlit/
   │   └── config.toml
   ├── app.py
   ├── requirements.txt
   └── README.md
   ```
2. Connect your GitHub account to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select your repository, branch, and specify `app.py` as the entrypoint file.
4. Click **Deploy**. Streamlit Cloud will read `requirements.txt` and configure the container environment.

### Docker Deployment
You can package this dashboard as a standalone container:

1. Create a `Dockerfile` in the root:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY . .
   RUN pip install --no-cache-dir -r requirements.txt
   EXPOSE 8501
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```
2. Build and run the image:
   ```bash
   docker build -t nexus-cnn .
   docker run -p 8501:8501 nexus-cnn
   ```

---

## 🧬 Underlying Neural Network Architecture
The app maps the following 9-layer CNN implemented in the project notebook:

| Layer | Type | Output Dimensions | Activation | Description |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Input | $150 \times 150 \times 3$ | - | Normalizes input RGB matrix values to $[0.0, 1.0]$. |
| 2 | Conv2D | $148 \times 148 \times 32$ | ReLU | 32 filters ($3 \times 3$ kernel) for edge/contrast patterns. |
| 3 | MaxPooling2D | $74 \times 74 \times 32$ | - | Spatial downsampling ($2 \times 2$ pool). |
| 4 | Conv2D | $72 \times 72 \times 64$ | ReLU | 64 filters ($3 \times 3$ kernel) for intermediate shapes. |
| 5 | MaxPooling2D | $36 \times 36 \times 64$ | - | Spatial downsampling ($2 \times 2$ pool). |
| 6 | Conv2D | $34 \times 34 \times 128$ | ReLU | 128 filters ($3 \times 3$ kernel) for abstract features. |
| 7 | MaxPooling2D | $17 \times 17 \times 128$ | - | Final spatial downsampling ($2 \times 2$ pool). |
| 8 | Flatten | $36992$ | - | Reshapes output matrix into 1D vector. |
| 9 | Dense | $128$ | ReLU | Fully connected hidden classification layer. |
| 10 | Dense (Out) | $1$ | Sigmoid | Target probability return value ($0$ to $1$). |
