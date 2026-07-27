# Galaxy Distance Classification using Machine Learning (Simulation)

**ACADEMIC INTEGRITY NOTICE** This repository contains the simulation source code for my ongoing undergraduate thesis in Physics at IPB University.

**Status:** Work in Progress (WIP)

**Repository Scope:** This repository contains the preliminary simulation, experimentation, and workflow development only. It does **not** represent the final implementation used in my undergraduate thesis.

**Usage:** This repository is published for educational and portfolio purposes only. Any form of plagiarism or unauthorized use for academic submissions is strictly prohibited.

**Publication:** A scientific publication based on the final research is planned upon completion of the thesis.

---

# Project Overview

This repository contains the simulation code and research workflow for my undergraduate thesis. The project focuses on estimating **photometric redshift (photo-z)** using machine learning techniques and utilizing the estimated redshift to classify galaxies into distance-based categories.

The ultimate objective of this research is to support astrophysical studies on galaxy evolution and cosmic epochs by providing an efficient alternative to spectroscopic observations.
# Galaxy Distance Classification using Machine Learning (Simulation)

---

# Research Objectives

The project consists of two main tasks:

### Photometric Redshift Estimation
Estimate galaxy redshifts from photometric observations using machine learning models.

### Galaxy Distance Classification
Classify galaxies into distance categories based on the predicted photometric redshift.

---

# Dataset

The project utilizes data from:

- Sloan Digital Sky Survey (SDSS) Data Release 16 (DR16)
- Main Galaxy Sample (MGS)
- BOSS
- eBOSS

The simulation dataset contains galaxies with spectroscopic redshifts that serve as the ground truth for model development and evaluation.

---

# Input Features

The current simulation utilizes several photometric and geometric features, including:

### Photometric Features
- SDSS model magnitudes
  - u
  - g
  - r
  - i
  - z
- Color indices
  - u − g
  - g − r
  - r − i
  - i − z

### Geometric Features
- Celestial coordinates
- Galaxy shape parameters
- Additional geometric descriptors

---

# Data Preprocessing

The simulation workflow includes:

- Data acquisition from SDSS DR16
- Data cleaning
- Missing value handling
- Feature engineering
- Color index generation
- Feature scaling
- Dataset splitting
- Model evaluation

---

# Machine Learning Workflow

The repository currently contains the experimental workflow for:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Machine learning model development
- Model evaluation
- Galaxy distance classification

> **Note:** The final machine learning architecture has **not yet been finalized** and will be implemented in the official undergraduate thesis repository.

---

# Tech Stack

### Programming Language
- Python

### Libraries
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Astropy
- Astroquery

### Development Environment
- Jupyter Notebook
- Google Colab

---

# Repository Status

This repository is intended as a **research simulation repository**.

It documents the experimental process, model exploration, and workflow development carried out prior to the implementation of the final undergraduate thesis model.

The final research code, optimized model, and complete experimental results may differ from the contents of this repository.

---

# Citation / Author

Created by **Najwa Aaliyah** Undergraduate Student Department of Physics IPB University

**- Email**: [najwapriyatna@gmail.com]
**- Linkedin:** [https://www.linkedin.com/in/najwapriyatna]
