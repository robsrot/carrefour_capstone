# Carrefour Data Challenge
End-to-end behavioral customer segmentation pipeline — discovering organic tribes from raw purchase ticket data.

---

## Prerequisites

Make sure you have **Conda** installed before starting.
If not, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/robsrot/carrefour_capstone.git
cd carrefour_capstone
```

### 2. Create the environment

```bash
conda env create -f environment.yml
```

This installs Python and all dependencies automatically.

### 3. Activate the environment

```bash
conda activate carrefour
```

You should see `(carrefour)` appear in your terminal.

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

This only needs to be done once. It makes the `carrefour` environment available inside Jupyter and VS Code.

### 5. Add your data files

Raw data files are not included in this repository.
Place the raw data files here:
- data/raw/maestra_articulos.csv
- data/raw/linea_tickets.csv