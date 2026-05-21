# Carrefour Data Challenge
End-to-end behavioral customer segmentation pipeline — discovering organic tribes from raw purchase ticket data.

## Prerequisites

Make sure you have **Conda** installed before starting.
If not, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) first.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/robsrot/carrefour_capstone.git
cd carrefour_capstone
```

### 2. Create the environment

```bash
conda env create -f environment.yml
conda activate carrefour
```

### 3. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=carrefour --display-name "Python (carrefour)"
```

This only needs to be done once. It makes the `carrefour` environment available inside Jupyter and VS Code.

### 4. Add your data files

Raw data files are not included in this repository.
Place the two raw CSV files in `data/raw/csv/`:

```
data/raw/csv/ie_maestra_articulos.csv
data/raw/csv/ie_linea_ticket.csv
```

### 5. Convert CSV to Parquet

Open `notebooks/01_exploration.ipynb` and run the **one-time setup** cell.
This converts the raw CSVs to Parquet for fast loading. Only needs to be done once.
The `ie_linea_ticket.csv` file is ~26 GB — conversion takes 5–15 minutes depending on your disk.
