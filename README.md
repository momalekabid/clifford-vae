### 1. Clone the repository

```sh
git clone https://github.com/momalekabid/clifford-vae.git
cd clifford-vae
```

---

### 2. Create and activate a virtual environment

```sh
# using uv (recommended)
uv venv .venv
source .venv/bin/activate

# using python directly
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

#### **A. With `uv` (Recommended)**

**Bash/Zsh:**
```sh
uv pip install -r <(uv pip compile pyproject.toml)
```

**Fish shell:**
```fish
uv pip compile pyproject.toml > requirements.tx
uv pip install -r requirements.txt
```

#### **B. With `pip`**

```sh
pip install -r requirements.txt
```

#### **C. Install the local `power_spherical` package**

```sh
uv pip install -e ./src/power_spherical
# or
pip install -e ./src/power_spherical
```

---

### 4. Run a Clifford/Hyperspherical VAE experiment

```sh
cd src
python clifford/mnist_clifford.py
```


## example setup for bash/zsh 

```sh
uv venv .venv
source .venv/bin/activate
uv pip install -e ./src/power_spherical
uv pip install -r <(uv pip compile pyproject.toml)
cd src
python clifford/mnist_clifford.py
```


## example setup (fish) 

```fish
uv venv .venv
source .venv/bin/activate
uv pip install -e ./src/power_spherical
uv pip compile pyproject.toml > requirements.txt
uv pip install -r requirements.txt
cd src
python clifford/mnist_clifford.py
```

---

## References

- [power_spherical github](https://github.com/nicola-decao/power_spherical)
- [uv documentation](https://github.com/astral-sh/uv)
- [original hyperspherical VAE paper](https://arxiv.org/abs/1804.00891)

