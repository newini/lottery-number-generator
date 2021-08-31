# Lottery expected next number generator


## 1. Instroduction



## 2. Preparation
- Install python packages
```
pip install -r requirements.txt
```



## How it works

- Read number data from csv file

- Calculate Probability on n+1

Detail is [here](https://newini.github.io/lottery-number-generator/docs/detail.html)



## JupyterLab extensions
**CAUTION**
Do not install `jupyterlab-kite` and `jupyterlab-lsp` both!


### 1. [jupyterlab-kite](https://github.com/kiteco/jupyterlab-kite)
to use full-auto completion

- Install `Kite` on user directory `~/.local/share`
```
bash -c "$(wget -q -O - https://linux.kite.com/dls/linux/current)"
```
- Install Kite Extension for JupyterLab
```
pip install jupyterlab-kite
```
- Enable server
```
jupyter server extension enable --user --py jupyter_kite
```
- Check
```
jupyter server extension list
```


### 2. [jupyterlab-lsp](https://github.com/krassowski/jupyterlab-lsp)
to use semi-auto completion
- Install
```
pip install jupyterlab-lsp
```
- Install language
```
pip install 'python-lsp-server[all]'
```
- Enable server
```
jupyter server extension enable --user --py jupyter_lsp
# jupyter labextension install @krassowski/jupyterlab-lsp
```
- Check
```
jupyter server extension list
```
