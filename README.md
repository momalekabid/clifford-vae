### 1. Clone the repository

```sh
git clone https://github.com/momalekabid/clifford-vae.git
cd clifford-vae
```

---

### 2. Create and activate a virtual environment

```sh
# using uv 
uv venv .venv
source .venv/bin/activate

# using python directly, if not using uv
python3 -m venv .venv
source .venv/bin/activate


# using conda (for SLURM jobs on WatGPU)
conda env create -f environment.yml
conda activate clifford-vae # if using SLURM/GPU cluster script, use source instead of conda
```

---

### 3. Install dependencies (ignore if using conda)
#### **A. With `uv` **

**Bash/Zsh:**
```sh
uv pip install -r <(uv pip compile pyproject.toml)
```

#### **B. With `uv` **

**Bash/Zsh:**
```sh
uv pip install -r <(uv pip compile pyproject.toml)
```

**Fish shell:**
```fish
uv pip compile pyproject.toml > requirements.tx
uv pip install -r requirements.txt
```


#### **C. Install the local `power_spherical` package**

```sh
cd src/power_spherical
uv pip install -e ./src/power_spherical
# OR (if using conda and/or venv)
pip install -e .
```

---

### 4. Run a Clifford/Hyperspherical VAE experiment

```sh
cd src
python clifford/mnist_clifford.py
```
---

## References

- [power_spherical github](https://github.com/nicola-decao/power_spherical)
- [uv documentation](https://github.com/astral-sh/uv)
- [original hyperspherical VAE paper](https://arxiv.org/abs/1804.00891)

