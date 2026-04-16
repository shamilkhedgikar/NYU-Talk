# RelWeights Lab Scaffold

This directory now holds a deployable, notebook-style lab scaffold for:

**From Overlays to Operators: Relational Spatial Weights for Data-Driven Policy**

The first pass uses **Jupyter Book** as the presentation layer because it gives us:

- sidebar navigation
- LaTeX math
- text-plus-code notebook pages
- embedded HTML and PDF blocks
- a clean path to Colab launch buttons once the source is published

## Structure

```text
lab/
  book/                  # Jupyter Book source
    _config.yml
    _toc.yml
    intro.md
    assets/
    content/
  notebooks/             # standalone analysis notebooks to add later
  src/                   # reusable Python operators/analysis code
  data/                  # raw/interim/processed inputs
  references/            # extra notes, citations, and imported materials
```

## Local build

```powershell
pip install -r lab/requirements.txt
jupyter-book build lab/book
```

## Colab wiring

The scaffold is organized so each module page can become a runnable notebook page.
Once this project is pushed to GitHub, we can turn on Jupyter Book `launch_buttons`
for Colab in `lab/book/_config.yml`.

