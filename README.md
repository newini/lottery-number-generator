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

### 1. [jupyterlab-lsp](https://github.com/krassowski/jupyterlab-lsp): to use semi-auto completion
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


### 2. [jupyterlab-kite](https://github.com/kiteco/jupyterlab-kite) (not work...)
to use auto completion

- Install `Kite`
```
bash -c "$(wget -q -O - https://linux.kite.com/dls/linux/current)"
```
- Install Kite Extension for JupyterLab
```
pip install jupyterlab-kite
```
(to be continued...)
