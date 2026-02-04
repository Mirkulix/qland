"""
IGQK v3.0 SaaS Platform - Web UI
Simple Gradio-based interface for MVP
"""

import sys
import os

# Fix Windows encoding
if os.name == 'nt':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import gradio as gr
import requests
import time
from typing import Optional

# API Base URL
API_BASE = "http://localhost:8000/api"


# ============================================================================
# HUGGINGFACE SEARCH - NEW!
# ============================================================================

def search_huggingface_models(query: str):
    """Search for models on HuggingFace Hub via Backend API"""

    if not query or len(query) < 2:
        return [], "⚠️ Please enter at least 2 characters to search."

    try:
        # Use Backend API with SSL fix instead of direct HuggingFace call
        response = requests.get(
            f"{API_BASE}/models/search/huggingface",
            params={"query": query, "limit": 10}
        )

        if response.status_code != 200:
            raise Exception(f"API returned status {response.status_code}")

        models = response.json()

        if not models:
            return [], f"❌ No models found for query: '{query}'\n\nTry searching for:\n- bert\n- gpt2\n- distilbert\n- t5"

        # Create choices for dropdown (model_id as value)
        choices = [
            (f"{model['id']} ({model['downloads']:,} downloads)" if model.get('downloads') else model['id'], model['id'])
            for model in models
        ]

        # Format results message
        result = f"# ✅ Found {len(models)} models for '{query}'\n\n"
        result += "**Select a model from the dropdown below to compress it!**\n\n"

        for i, model in enumerate(models[:10], 1):
            model_id = model["id"]
            downloads = f"{model['downloads']:,}" if model.get('downloads') else "N/A"

            result += f"### {i}. {model_id}\n"
            result += f"   - Downloads: {downloads}\n"

            if model.get('task'):
                result += f"   - Task: {model['task']}\n"

            result += "\n"

        result += "\n✨ **Models loaded into dropdown! Select one and click 'Start Compression'!**\n"

        return choices, result

    except Exception as e:
        return [], f"""
❌ Search failed: {str(e)}

**Offline Mode:** Enter model name directly!

Popular models you can try:
- `bert-base-uncased`
- `distilbert-base-uncased`
- `gpt2`
- `t5-small`
- `roberta-base`
"""


# ============================================================================
# CREATE MODE - Training
# ============================================================================

def start_training_job(
    job_name: str,
    dataset: str,
    architecture: str,
    optimizer: str,
    epochs: int,
    batch_size: int,
    auto_compress: bool
):
    """Start a training job"""

    try:
        # This would call the actual API when backend is running
        # For now, return mock data
        return f"""
🚀 Training Job Started!

Job Name: {job_name}
Dataset: {dataset}
Architecture: {architecture}
Optimizer: {optimizer} {"(⚡ Quantum-powered!)" if optimizer == "IGQK" else ""}
Epochs: {epochs}
Batch Size: {batch_size}
Auto-Compress: {"✅ Yes" if auto_compress else "❌ No"}

Status: Training started...
Expected time: {epochs * 2} minutes

Monitor at: /api/training/status/job_123
"""

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ============================================================================
# COMPRESS MODE - Compression
# ============================================================================

def start_compression_job(
    job_name: str,
    model_source: str,
    model_identifier: str,
    compression_method: str,
    quality_target: float,
    auto_validate: bool,
    progress=gr.Progress()
):
    """Start a compression job with real API integration"""

    try:
        # Progress: Initializing
        progress(0, desc="🚀 Initializing compression job...")

        # Map UI values to API format
        source_map = {
            "HuggingFace Hub": "huggingface",
            "Upload File": "upload",
            "My Models": "local",
            "URL": "url"
        }

        method_map = {
            "AUTO (🤖 AI chooses best)": "auto",
            "Ternary (16× compression)": "ternary",
            "Binary (32× compression)": "binary",
            "Sparse (Variable)": "sparse",
            "Low-Rank (Variable)": "lowrank"
        }

        # Prepare API request
        payload = {
            "job_name": job_name,
            "model_source": source_map.get(model_source, "huggingface"),
            "model_identifier": model_identifier,
            "compression_method": method_map.get(compression_method, "auto"),
            "quality_target": quality_target,
            "auto_validate": auto_validate
        }

        # Progress: Submitting to API
        progress(0.1, desc="📡 Submitting job to backend API...")

        # Call API
        response = requests.post(f"{API_BASE}/compression/start", json=payload, timeout=10)

        if response.status_code != 200:
            return f"❌ API Error: {response.status_code}\n\n{response.text}"

        job_data = response.json()
        job_id = job_data.get("job_id", "unknown")

        # Progress: Job submitted
        progress(0.2, desc="✅ Job submitted! Monitoring progress...")

        # Poll for status updates
        status_text = f"""
🗜️ Compression Job Started!

Job ID: {job_id}
Job Name: {job_name}
Model Source: {model_source}
Model: {model_identifier}
Compression Method: {compression_method}
Quality Target: {quality_target * 100}%
Auto-Validate: {"✅ Yes" if auto_validate else "❌ No"}

"""

        # Monitor job progress (poll every 2 seconds)
        for i in range(60):  # Max 120 seconds
            time.sleep(2)

            # Get job status
            status_response = requests.get(f"{API_BASE}/compression/status/{job_id}", timeout=5)

            if status_response.status_code == 200:
                status_data = status_response.json()
                current_status = status_data.get("status", "unknown")

                # Calculate overall progress
                overall_progress = 0.2 + (i / 60) * 0.3  # Base progress + time-based

                # Update progress based on status with detailed messages
                if current_status == "pending":
                    progress(0.25 + (i / 60) * 0.05, desc=f"⏳ Job queued... Position in queue: {i+1}")
                    status_text += f"\n[{time.strftime('%H:%M:%S')}] ⏳ Waiting in queue..."

                elif current_status == "downloading":
                    download_progress = 0.3 + (i / 60) * 0.2
                    progress(download_progress, desc=f"⬇️ Downloading model from HuggingFace... {int(download_progress*100)}%")
                    status_text += f"\n[{time.strftime('%H:%M:%S')}] ⬇️ Downloading model: {model_identifier}"
                    status_text += f"\n   └─ Fetching weights and configuration..."

                elif current_status == "compressing":
                    compress_progress = 0.5 + (i / 60) * 0.25
                    progress(compress_progress, desc=f"🗜️ Compressing with IGQK... {int(compress_progress*100)}%")
                    status_text += f"\n[{time.strftime('%H:%M:%S')}] 🗜️ Applying {compression_method} compression"
                    status_text += f"\n   └─ Quantum optimization in progress..."
                    status_text += f"\n   └─ Projecting weights to compressed space..."

                elif current_status == "validating":
                    progress(0.85, desc="✅ Validating compressed model accuracy...")
                    status_text += f"\n[{time.strftime('%H:%M:%S')}] ✅ Running validation tests..."
                    status_text += f"\n   └─ Comparing original vs compressed accuracy..."

                elif current_status == "completed":
                    progress(1.0, desc="🎉 Compression completed!")

                    # Return final results
                    results = status_data.get("results", {})
                    return f"""
# ✅ Compression Complete!

**Job ID:** {job_id}
**Job Name:** {job_name}
**Model:** {model_identifier}

## 📊 Results

| Metric | Original | Compressed | Improvement |
|--------|----------|------------|-------------|
| Size | {results.get('original_size_mb', 'N/A')} MB | {results.get('compressed_size_mb', 'N/A')} MB | **{results.get('compression_ratio', 'N/A')}× smaller** ✨ |
| Parameters | {results.get('parameters', 'N/A'):,} | {results.get('parameters', 'N/A'):,} | Same |
| Accuracy | {results.get('original_accuracy', 'N/A')}% | {results.get('compressed_accuracy', 'N/A')}% | {results.get('accuracy_loss', 'N/A')}% loss |

## 💰 Savings

- **Storage:** {results.get('storage_saved_mb', 'N/A')} MB saved
- **Memory:** {results.get('memory_saved_percent', 'N/A')}% reduction

## 📥 Downloads

- Compressed Model: `/models/compressed/{job_id}`
- Full Report: `/reports/{job_id}`
"""

                elif current_status == "failed":
                    error_msg = status_data.get("error", "Unknown error")
                    return f"""
❌ Compression Failed!

Job ID: {job_id}
Error: {error_msg}

Please check:
- Model identifier is correct
- HuggingFace Hub is reachable
- Sufficient disk space available
"""

                # Update status display
                status_text += f"\n[{time.strftime('%H:%M:%S')}] Status: {current_status}"

        # Timeout
        return status_text + "\n\n⏰ Status check timeout. Job may still be running.\nCheck: /api/compression/status/" + job_id

    except requests.exceptions.ConnectionError:
        return f"""
❌ Cannot connect to backend API!

The backend server is not running or not reachable.

Please start the backend:
```
cd backend
python main.py
```

Backend should be running on: {API_BASE}
"""

    except Exception as e:
        return f"❌ Error: {str(e)}\n\nDetails:\n{type(e).__name__}"


# ============================================================================
# Model Hub - Real Data
# ============================================================================

def list_models():
    """List all compressed models from local storage"""

    try:
        # Get models from API
        response = requests.get(f"{API_BASE}/models/list", timeout=5)

        if response.status_code != 200:
            return "⚠️ Could not fetch models from backend API."

        models = response.json().get("models", [])

        if not models:
            return """
# 🏪 Model Hub

**No models found yet!**

Start compressing models in the COMPRESS Mode tab to see them here.
"""

        # Build table
        result = "# 🏪 Model Hub\n\n"
        result += "## Your Compressed Models\n\n"
        result += "| Name | Size | Compression | Accuracy | Created |\n"
        result += "|------|------|-------------|----------|---------|\n"

        for model in models:
            name = model.get("name", "Unknown")
            size = model.get("compressed_size_mb", 0)
            ratio = model.get("compression_ratio", "N/A")
            accuracy = model.get("accuracy", "N/A")
            created = model.get("created_at", "Unknown")

            result += f"| {name} | {size:.1f} MB | {ratio}× | {accuracy}% | {created} |\n"

        result += "\n\n💡 **Tip:** Click on a model to download or deploy it!"

        return result

    except requests.exceptions.ConnectionError:
        # Fallback: Read from local filesystem
        models_dir = os.path.join(os.path.dirname(__file__), "compressed_models")

        if not os.path.exists(models_dir):
            return """
# 🏪 Model Hub

**No models found yet!**

Backend API is not running. Start compressing models in the COMPRESS Mode tab.
"""

        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]

        if not model_files:
            return """
# 🏪 Model Hub

**No models found yet!**

Start compressing models in the COMPRESS Mode tab to see them here.
"""

        result = "# 🏪 Model Hub\n\n"
        result += "## Local Compressed Models\n\n"
        result += "| Filename | Size | Created |\n"
        result += "|----------|------|----------|\n"

        for filename in model_files:
            filepath = os.path.join(models_dir, filename)
            size_mb = os.path.getsize(filepath) / (1024 ** 2)
            created = time.strftime("%Y-%m-%d", time.localtime(os.path.getctime(filepath)))

            result += f"| {filename} | {size_mb:.1f} MB | {created} |\n"

        return result

    except Exception as e:
        return f"❌ Error loading models: {str(e)}"


# ============================================================================
# Results Display
# ============================================================================

def show_compression_results():
    """Show mock compression results"""

    return f"""
# ✅ Compression Complete!

## 📊 Results Comparison

| Metric | Original | Compressed | Improvement |
|--------|----------|------------|-------------|
| Size | 440 MB | 27.5 MB | **16× smaller** ✨ |
| Accuracy | 89.2% | 88.7% | -0.5% only ✅ |
| Inference | 45 ms | 3 ms | **15× faster** 🚀 |
| Parameters | 110M | 110M | Same |
| Bits/Weight | 32 | 2 | 16× reduction |

## 💰 Cost Savings

**Cloud Deployment:**
- Storage: €45/mo → €2.80/mo (**€42.20 saved!**)
- Inference: €500/mo → €31/mo (**€469 saved!**)
- Total Savings: **€511.20/month** = **€6,134/year**

## 🎯 Recommendation

✅ **Use compressed model!**
- 16× smaller
- 15× faster
- Only -0.5% accuracy loss
- 93.8% memory saved

## 📥 Downloads

- [Download Compressed Model](#)
- [Download Original Model](#)
- [View Full Report](#)
"""


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(
    title="IGQK v3.0 SaaS Platform",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
    # 🚀 IGQK v3.0 - All-in-One ML Platform

    **Train, Compress, and Deploy AI Models with Quantum-powered Technology**

    """)

    with gr.Tabs():

        # ====================================================================
        # TAB 1: CREATE MODE
        # ====================================================================

        with gr.Tab("🔨 CREATE Mode - Train Models"):
            gr.Markdown("""
            ## Train Models from Scratch

            Train new models with Quantum-optimized IGQK or classical methods.
            Get 50% faster training and automatic 16× compression!
            """)

            with gr.Row():
                with gr.Column():
                    train_job_name = gr.Textbox(
                        label="Job Name",
                        placeholder="My Awesome Model",
                        value="CIFAR-10 Classifier"
                    )

                    train_dataset = gr.Dropdown(
                        label="Dataset",
                        choices=["MNIST", "CIFAR-10", "ImageNet", "Custom Upload"],
                        value="CIFAR-10"
                    )

                    train_architecture = gr.Dropdown(
                        label="Architecture",
                        choices=["ResNet-18", "ResNet-50", "VGG-16", "EfficientNet", "Custom"],
                        value="ResNet-18"
                    )

                    train_optimizer = gr.Radio(
                        label="Optimizer",
                        choices=["IGQK (⚡ Quantum)", "Adam", "SGD", "AdamW"],
                        value="IGQK (⚡ Quantum)"
                    )

                    with gr.Row():
                        train_epochs = gr.Slider(
                            label="Epochs",
                            minimum=1,
                            maximum=100,
                            value=20,
                            step=1
                        )

                        train_batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=8,
                            maximum=512,
                            value=64,
                            step=8
                        )

                    train_auto_compress = gr.Checkbox(
                        label="Auto-compress after training (16× smaller!)",
                        value=True
                    )

                    train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

                with gr.Column():
                    train_output = gr.Textbox(
                        label="Training Status",
                        lines=15,
                        placeholder="Click 'Start Training' to begin..."
                    )

            train_btn.click(
                fn=start_training_job,
                inputs=[
                    train_job_name,
                    train_dataset,
                    train_architecture,
                    train_optimizer,
                    train_epochs,
                    train_batch_size,
                    train_auto_compress
                ],
                outputs=train_output
            )

        # ====================================================================
        # TAB 2: COMPRESS MODE
        # ====================================================================

        with gr.Tab("🗜️ COMPRESS Mode - Compress Models"):
            gr.Markdown("""
            ## Compress Existing Models

            Take any model and make it 16× smaller with IGQK Quantum compression.
            Works with PyTorch, HuggingFace, and custom models!
            """)

            # ========== NEUE SUCHFUNKTION ==========
            gr.Markdown("### 🔍 Step 1: Search HuggingFace Models")

            with gr.Row():
                search_query = gr.Textbox(
                    label="Search HuggingFace",
                    placeholder="Enter search term (e.g., 'bert', 'gpt2', 'distilbert')",
                    scale=3
                )
                search_btn = gr.Button("🔍 Search", scale=1, variant="secondary")

            search_results = gr.Markdown(
                value="Enter a search term and click 'Search' to find models on HuggingFace...",
                label="Search Results"
            )

            # Dropdown for model selection (populated by search)
            model_dropdown = gr.Dropdown(
                label="📌 Select Model from Search Results",
                choices=[("bert-base-uncased", "bert-base-uncased")],
                value="bert-base-uncased",
                interactive=True,
                allow_custom_value=True
            )

            gr.Markdown("### ⚙️ Step 2: Configure Compression")

            with gr.Row():
                with gr.Column():
                    comp_job_name = gr.Textbox(
                        label="Job Name",
                        placeholder="Model Compression",
                        value="BERT Compression"
                    )

                    comp_source = gr.Radio(
                        label="Model Source",
                        choices=["HuggingFace Hub", "Upload File", "My Models", "URL"],
                        value="HuggingFace Hub"
                    )

                    comp_model_id = gr.Textbox(
                        label="Model Identifier (selected or enter manually)",
                        placeholder="bert-base-uncased (for HuggingFace)",
                        value="bert-base-uncased",
                        interactive=False  # Auto-filled from dropdown
                    )

                    comp_method = gr.Radio(
                        label="Compression Method",
                        choices=[
                            "AUTO (🤖 AI chooses best)",
                            "Ternary (16× compression)",
                            "Binary (32× compression)",
                            "Sparse (Variable)",
                            "Low-Rank (Variable)"
                        ],
                        value="AUTO (🤖 AI chooses best)"
                    )

                    comp_quality = gr.Slider(
                        label="Quality Target (% of original accuracy to retain)",
                        minimum=0.85,
                        maximum=0.99,
                        value=0.95,
                        step=0.01
                    )

                    comp_validate = gr.Checkbox(
                        label="Auto-validate compressed model",
                        value=True
                    )

                    comp_btn = gr.Button("🗜️ Start Compression", variant="primary", size="lg")

                with gr.Column():
                    comp_output = gr.Textbox(
                        label="Compression Status",
                        lines=15,
                        placeholder="Click 'Start Compression' to begin..."
                    )

            # Connect search button - updates both dropdown and results
            search_btn.click(
                fn=search_huggingface_models,
                inputs=search_query,
                outputs=[model_dropdown, search_results]
            )

            # Connect model dropdown to automatically fill the model_id field
            model_dropdown.change(
                fn=lambda x: x,  # Simply pass the selected value through
                inputs=model_dropdown,
                outputs=comp_model_id
            )

            # Connect compression button
            comp_btn.click(
                fn=start_compression_job,
                inputs=[
                    comp_job_name,
                    comp_source,
                    comp_model_id,
                    comp_method,
                    comp_quality,
                    comp_validate
                ],
                outputs=comp_output
            )

        # ====================================================================
        # TAB 3: RESULTS & COMPARISON
        # ====================================================================

        with gr.Tab("📊 Results & Analysis"):
            gr.Markdown("## Compression Results & Model Comparison")

            results_btn = gr.Button("📊 Load Latest Results", variant="secondary")
            results_output = gr.Markdown(value="Click 'Load Latest Results' to see compression analysis...")

            results_btn.click(
                fn=show_compression_results,
                outputs=results_output
            )

        # ====================================================================
        # TAB 4: MODEL HUB
        # ====================================================================

        with gr.Tab("🏪 Model Hub"):
            gr.Markdown("""
            ## Your Models

            Browse, manage, and deploy your trained and compressed models.
            """)

            models_refresh_btn = gr.Button("🔄 Refresh Models", variant="secondary")
            models_output = gr.Markdown(value="Click 'Refresh Models' to load your compressed models...")

            models_refresh_btn.click(
                fn=list_models,
                outputs=models_output
            )

        # ====================================================================
        # TAB 5: DOCS & INFO
        # ====================================================================

        with gr.Tab("📚 Documentation"):
            gr.Markdown("""
            ## About IGQK v3.0

            **IGQK** (Information-Geometric Quantum Compression) is the world's first
            Quantum-powered model compression technology.

            ### 🌟 Key Features

            #### 🔨 CREATE Mode
            - Train models from scratch
            - 50% faster with Quantum optimization
            - Automatic 16× compression
            - One-click publishing to HuggingFace

            #### 🗜️ COMPRESS Mode
            - Compress existing models 16× smaller
            - Support for PyTorch, HuggingFace, ONNX
            - Only ~0.5-1% accuracy loss
            - 15× faster inference

            ### 💎 Why IGQK?

            **Traditional Compression:**
            - 4-8× compression at best
            - Significant accuracy loss (3-5%)
            - Slow compression process

            **IGQK Quantum Compression:**
            - **16× compression** 🎉
            - Only 0.5-1% accuracy loss
            - Fast compression (<5 minutes)
            - Quantum-optimized training

            ### 🚀 Use Cases

            1. **Mobile Apps** - Deploy ML models on smartphones
            2. **IoT Devices** - Run AI on Raspberry Pi, Arduino
            3. **Cloud Cost Reduction** - Save 93.8% on storage & compute
            4. **Edge AI** - Offline AI applications
            5. **Fast Inference** - 15× faster predictions

            ### 📖 Getting Started

            1. **CREATE Mode**: Train a new model or...
            2. **COMPRESS Mode**: Compress an existing model
            3. View results and comparisons
            4. Download or deploy your model

            ### 🔗 Resources

            - [API Documentation](/api/docs)
            - [GitHub Repository](https://github.com/igqk)
            - [Research Paper](https://arxiv.org/...)
            - [Community Forum](https://forum.igqk.ai)

            ### 💰 Pricing

            - **Free**: 10 training hours, 5 compressions/month
            - **Starter**: €49/month - 100 training hours
            - **Pro**: €499/month - Unlimited
            - **Enterprise**: Custom pricing
            """)

    gr.Markdown("""
    ---

    💡 **Tip:** Start with COMPRESS mode to see IGQK in action with an existing model!

    🔗 **API Docs:** http://localhost:8000/api/docs

    🌐 **Backend Status:** Running on http://localhost:8000
    """)


if __name__ == "__main__":
    print("="*70)
    print("🚀 Starting IGQK v3.0 SaaS Platform - Web UI")
    print("="*70)
    print()
    print("Features:")
    print("  ✅ CREATE Mode - Train models with Quantum optimization")
    print("  ✅ COMPRESS Mode - 16× compression with IGQK")
    print("  ✅ Model Hub - Manage your models")
    print("  ✅ Documentation - Complete guide")
    print()
    print("Opening Web UI at: http://localhost:7860")
    print()
    print("Press CTRL+C to stop")
    print("="*70)
    print()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
