# ResNet-18 Image Classification & Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange.svg)

An advanced image classification web application built with Python, PyTorch, and Gradio. This tool leverages a pre-trained **ResNet-18** model (fine-tuned on **ImageNet**) to perform real-time inference on user-uploaded images.

The dashboard provides not just the top-k predictions, but also detailed analytics, model confidence metrics, and feature vector visualizations in a custom dark-neon interface.


## ✨ Key Features

* **Multi-Class Classification:** Identifies the top predictions from 1000 ImageNet classes.
* **Confidence Charting:** Dynamically generates a Matplotlib bar plot for top-5 class probabilities.
* **Full Data Export:** Provides all 1000 class probabilities in a sortable, downloadable `.csv` file.
* **Deep Model Analytics:** Calculates and displays key confidence metrics:
    * **Prediction Entropy:** Measures the model's uncertainty (a lower entropy value signifies higher certainty).
    * **Max Confidence:** The highest probability assigned to a single class.
    * **Top-5 Mean Confidence:** The average confidence across the top 5 predictions.
* **Feature Vector Visualization:** Extracts the 512-dimension feature vector from the penultimate layer and plots its distribution histogram.
* **Image Statistics:** Calculates and displays basic image properties (size, mode, RGB channel means/stddevs).

---

## 🔬 Technical Architecture & Jargon

This project integrates several key components in the MLOps and computer vision stack:

### 1. Model & Inference
* **CNN (Convolutional Neural Network):** Utilizes **ResNet-18**, a deep residual network architecture with 18 layers, known for its effective use of skip-connections to prevent vanishing gradients.
* **Transfer Learning:** Employs a model **pre-trained** on the **ImageNet (ILSVRC)** dataset, leveraging its learned hierarchical features for general-purpose classification.
* **Model Inference:** The model is set to evaluation mode (`model.eval()`) to disable **dropout** and **batch normalization** updates, ensuring deterministic output.
* **Device Agnostic:** Automatically selects `torch.device` ("cuda" or "cpu") based on available hardware (GPU/CPU).

### 2. Data Processing Pipeline
* **Preprocessing Transforms:** Input images undergo a sequential **torchvision transforms** pipeline ( `Resize`, `CenterCrop`, `ToTensor`, `Normalize`) to match the exact input requirements (224x224) and statistical distribution of the ImageNet-trained model.
* **Normalization:** Tensors are normalized using the specific ImageNet mean `[0.485, 0.456, 0.406]` and stddev `[0.229, 0.224, 0.225]` to center the data.

### 3. Output & Analytics
* **Logits to Probabilities:** The raw, unbounded output **logits** from the final fully-connected layer are converted into a probability distribution (summing to 1.0) using the **Softmax** activation function.
* **Feature Extraction:** A new `torch.nn.Sequential` model is created from the ResNet's children, effectively tapping the **penultimate layer** (the average pooling layer) to extract the 512-dimension **feature embedding**. This latent space representation is the model's high-level abstract understanding of the image.

### 4. Application Interface
* **UI Framework:** Built with **Gradio**, a high-level library for creating interactive machine learning web apps.
* **Component-Based UI:** The interface is constructed using `gr.Blocks`, allowing for a complex, custom layout of composable components (`gr.Image`, `gr.Dataframe`, `gr.Plot`, `gr.Label`).
* **Custom Styling:** Injects custom **CSS** to create a 'Dark Neon' theme, overriding default Gradio styles for a unique visual identity.

---

## 🚀 Installation & Local Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
    cd YOUR-REPO-NAME
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on MacOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    (Make sure your `requirements.txt` file is up-to-date)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python imageclassification.py
    ```

5.  **Open the app:**
    Navigate to the local URL shown in your terminal (usually `http://127.0.0.1:7860`).
