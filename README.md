# Object-Recognition-for-Automated-Billing-in-shopping-mart

## About
1.The Automatic Billing System uses deep learning and computer vision to automate product identification and billing.
2.It eliminates manual barcode scanning, improving efficiency and reducing errors.
3.The system helps retailers optimize inventory and personalize promotions through data analysis.
4.Designed to enhance customer experience and operational efficiency in retail.

## Prerequisites
Ensure the following tools and dependencies are installed:
- Python 3.9.4 or higher
- pytorch
- pillow
- mysql-connector-python
- tkinter

Verify your Python version:
```bash
python --version  # Output: Python 3.9.4
```

Create and Activate Virtual Environment
1.Create a virtual environment:
```bash
python -m venv venv
```
2.Activate the virtual environment:
On Windows:
```bash
venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```

Install pytorch:

Install PyTorch with GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 #for GPU
```

Alternatively, install PyTorch for CPU-only systems:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  #for cpu
```

```bash
pip install pillow
pip install torch  
pip install mysql-connector-python
```

### create 1 annotation file for ur dataset 
to create, run the generate_annotations.py code
path dataset->images->annotations.csv


## steps
1. run train_model.py
  ```bash
  python train_model.py
  ```
2. run model.py
  ```bash
  python model.py
  ```
3. run main.py (MAIN FILE)
   ```bash
   python main.py
   ```
   (this is the main file linked to tkinter(frontend))

IMP:
The list of products in labels.txt should match with main.py code
also you can get the keras_model.h5 from https://teachablemachine.withgoogle.com/
this is the application where you can train your model (A fast, easy way to create machine learning models)

## Contact
Varnitha Raghunath
- https://linkedin.com/in/varnitha-raghunath-5b3a072ab 
- varnitharaghunath.varu14@gmail.com
