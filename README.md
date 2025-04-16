# malaria cnn

# src images (unzip images)

- https://github.com/We-Gold/Malaria

# Install Miniconda

- Download and install Miniconda from the website: [Download Miniconda](https://docs.anaconda.com/miniconda/)

# Create a conda - torch environment

```
conda create --name torch31111 python=3.11.11
```

# Activate the environment

```
conda activate torch31111
```

# Install dependencies

```
pip install -r requirements.txt
```

# Run / Test Inference

```
python app.py
```

# Install dependencies manually (optional)

```
conda install -y flask
conda install -y Flask-WTF
conda install -y joblib
conda install -y scikit-learn
pip install scikit-learn==1.6.1
conda install -y scipy
conda install -y werkzeug
conda install -y pandas
conda install -y numpy
conda install -y seaborn
conda install -y matplotlib
conda install -y pickle
conda install -y pytorch torchvision torchaudio cpuonly -c pytorch
conda install -y sklearn
conda install -y gunicorn
```