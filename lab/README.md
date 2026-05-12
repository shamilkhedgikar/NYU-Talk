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

Lab 3 reads its cached Data Commons extract during normal builds. Set `DATACOMMONS_API_KEY` or `DC_API_KEY` only when you want to refresh that cache.

```powershell
pip install -r lab/requirements.txt
Set-Location lab/book
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=relweights_lab content/05-lab-3-interpolation-with-relweights/interpolation.ipynb
python sync_nav.py
python -m jupyter_book build --html
python postprocess_site.py
python -m jupyter_book start
```

## Colab wiring

The scaffold is organized so each module page can become a runnable notebook page.
Once this project is pushed to GitHub, we can turn on Jupyter Book `launch_buttons`
for Colab in `lab/book/_config.yml`.

