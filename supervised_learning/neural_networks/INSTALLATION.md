# Installation Guide - Neurons and Layers Demo

## ⚠️ Important: Choose the Right Method for Your System

This guide provides multiple installation methods. Choose the one that works best for your situation.

---

## 🎯 Method 1: Virtual Environment (Recommended for Development)

**Best for:** Clean isolated environment, testing, development

### Step 1: Create Virtual Environment

```bash
cd supervised_learning/neural_networks

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install packages one by one (better for slow connections)
pip install numpy
pip install matplotlib
pip install tensorflow

# OR use requirements file
pip install -r requirements.txt
```

### Step 3: Run the Script

```bash
python neurons_and_layers_demo.py
```

### Step 4: Deactivate When Done

```bash
deactivate
```

---

## 🐧 Method 2: System Packages (Recommended for Debian/Ubuntu/Parrot OS)

**Best for:** Externally-managed Python environments, avoiding large downloads

### Install via APT

```bash
# Update package list
sudo apt update

# Install dependencies
sudo apt install -y python3-numpy python3-matplotlib python3-tensorflow

# Or install everything at once
sudo apt install -y python3-full python3-numpy python3-matplotlib python3-tensorflow python3-tk
```

### Run the Script

```bash
cd supervised_learning/neural_networks
python3 neurons_and_layers_demo.py
```

**Advantages:**
- ✅ No virtual environment needed
- ✅ Faster installation (uses system packages)
- ✅ Managed by system package manager
- ✅ Works with externally-managed Python environments

**Disadvantages:**
- ❌ May not have latest versions
- ❌ System-wide installation

---

## 🚀 Method 3: Automated Setup Script

**Best for:** Quick automated setup with virtual environment

### Run the Script

```bash
cd supervised_learning/neural_networks
bash setup.sh
```

The script will:
1. Check for Python 3
2. Create virtual environment
3. Install all dependencies
4. Verify installation

**Note:** May fail with slow/unstable internet connections (TensorFlow is 620 MB)

---

## 💡 Method 4: Lightweight Installation (Without TensorFlow)

**If you just want to understand the code without running it:**

### Install Only Documentation Tools

```bash
# In virtual environment
source venv/bin/activate
pip install numpy matplotlib

# Or system-wide
sudo apt install python3-numpy python3-matplotlib
```

### Run Without TensorFlow

You can still read the code and understand the concepts. To run a simplified version:

```python
# Comment out TensorFlow imports at the top of neurons_and_layers_demo.py
# import tensorflow as tf  # Comment this
# from tensorflow.keras.models import Sequential  # Comment this
# from tensorflow.keras.layers import Dense, Input  # Comment this
```

---

## 🌐 Method 5: Alternative TensorFlow Installation

**For users with network issues:**

### Option A: Install CPU-only version (smaller)

```bash
pip install tensorflow-cpu
```

### Option B: Download manually

```bash
# Download from mirror or cache
# Visit: https://pypi.org/project/tensorflow/#files
# Download the appropriate .whl file for your Python version
# Then install locally:
pip install /path/to/downloaded/tensorflow-2.20.0-*.whl
```

### Option C: Use conda (if available)

```bash
conda install -c conda-forge tensorflow
```

---

## 🔍 Troubleshooting

### Error: "externally-managed-environment"

**Solution:** Use virtual environment (Method 1) or system packages (Method 2)

```bash
# Virtual environment approach
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Error: "Network is unreachable" or download timeouts

**Solution:** Use system packages (Method 2)

```bash
sudo apt install python3-tensorflow
```

### Error: "No module named 'tensorflow'"

**Check if in virtual environment:**
```bash
which python
# Should show: .../venv/bin/python if in venv
```

**Check if TensorFlow installed:**
```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

**Install if missing:**
```bash
# Via apt
sudo apt install python3-tensorflow

# Or via pip in venv
pip install tensorflow
```

### Display Issues (plots don't show)

```bash
# Install Tk backend
sudo apt install python3-tk

# Or in virtual environment
pip install tk
```

### Permission Denied

```bash
# Make setup script executable
chmod +x setup.sh
```

---

## ✅ Verification

After installation, verify everything works:

```bash
# Check Python version
python3 --version

# Check NumPy
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Check Matplotlib
python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"

# Check TensorFlow
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
```

Expected output:
```
NumPy: 2.x.x
Matplotlib: 3.x.x
TensorFlow: 2.x.x
```

---

## 📦 Quick Reference

| Method | Command | Best For |
|--------|---------|----------|
| **Virtual Env** | `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt` | Development, isolation |
| **System Packages** | `sudo apt install python3-numpy python3-matplotlib python3-tensorflow` | Debian/Ubuntu/Parrot OS |
| **Auto Setup** | `bash setup.sh` | Quick automated setup |
| **Manual** | `pip install numpy matplotlib tensorflow` | Custom installations |

---

## 🎓 Recommended for Parrot OS / Debian Users

**Best approach (in order of preference):**

1. **System packages** - Fast, no virtual env needed
   ```bash
   sudo apt install python3-tensorflow python3-matplotlib python3-numpy
   python3 neurons_and_layers_demo.py
   ```

2. **Virtual environment** - Clean, isolated
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install numpy matplotlib tensorflow
   python neurons_and_layers_demo.py
   ```

---

## 📝 Next Steps

After successful installation:
1. Run the demo: `python3 neurons_and_layers_demo.py`
2. Read the output and follow the interactive prompts
3. View the visualizations
4. Modify the code to experiment
5. Read [README_neurons_layers.md](README_neurons_layers.md) for detailed documentation

---

**Need help?** Check [QUICKSTART.md](QUICKSTART.md) for usage examples.

**Happy Learning! 🧠🚀**
