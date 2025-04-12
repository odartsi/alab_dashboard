# 🧪 ALab Interactive Dashboard

An interactive dashboard developed using **Dash** and **Plotly** to support material discovery workflows at Lawrence Berkeley National Lab.  
It brings together synthesis and characterization data in one place to help researchers:

- Compare experimental samples
- Identify trends or anomalies
- Explore ideas for new analyses

---

## 📊 Dashboard Demo

🎥 **Watch the video demo**:  

https://github.com/user-attachments/assets/fdf3d33a-c34c-4738-9f2b-3924d3544bac

---

## 🔐 Environment Setup & Installation

To keep credentials secure, this project uses environment variables loaded from a `.env` file.

### Step 1: Clone the repo

```bash
git clone https://github.com/your-username/ALAB_DASHBOARD_NEW.git
cd ALAB_DASHBOARD_NEW
```

### Step 2: Create a `.env` file

Copy the template:

```bash
cp .env.example .env
```

Then open `.env` and fill in your MongoDB credentials:

```env
SAURON_DB_USER=your_sauron_username
SAURON_DB_PASS=your_sauron_password
SAURON_DB_HOST=your_sauron_host
SAURON_DB_NAME=your_sauron_database

DARA_DB_USER=your_dara_username
DARA_DB_PASS=your_dara_password
DARA_DB_HOST=your_dara_host
DARA_DB_NAME=your_dara_database
DARA_COLLECTION=results
```

> 🔒 **Note**: The `.env` file is ignored via `.gitignore` and will not be tracked by git.

### Step 3: Set up a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App

Once the environment is set up:

```bash
python app.py
```

Then open your browser and go to:  
[http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 📁 Project Structure

```
├── app.py                # Entry point for the dashboard
├── data.py               # MongoDB access and caching
├── layout.py             # Layout and UI components
├── callbacks.py          # App interactivity
├── utils.py              # Helper functions
├── datasets/             # Local CSV cache
├── assets/               # Dash assets (e.g., custom CSS)
├── .env.example          # Template for secrets
├── requirements.txt      # Dependencies
└── README.md
```

---


