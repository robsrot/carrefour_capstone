# Carrefour Data Challenge
End-to-end behavioral customer segmentation pipeline — discovering organic tribes from raw purchase ticket data.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/robsrot/carrefour_capstone.git
cd carrefour_capstone
```

### 2. Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` appear at the start of your terminal line — this means the environment is active.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the data

Raw data files are not included in this repository. You must place the files here:
- data/raw/maestra_articulos.csv
- data/raw/linea_tickets.csv