# Lottery expected next number generator
Just for Fun



## 1. Preparation
- Install python packages
```
pip install -r requirements.txt
```


## 2. Usage
### 2.1 Open Jupyter Lab
```
jupyter lab --no-browser
```


### 2.2 Select book file
Jupyter book files is in `books`




## 3. JupyterLab extensions
**CAUTION**
Do not install `jupyterlab-kite` and `jupyterlab-lsp` both!


### 3.1 [jupyterlab-kite](https://github.com/kiteco/jupyterlab-kite)
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


### 3.2 [jupyterlab-lsp](https://github.com/krassowski/jupyterlab-lsp)
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



## 4. How it works

- Read number data from csv file

- Calculate Probability on n+1

Detail is [here](https://newini.github.io/lottery-number-generator/docs/detail.html)
