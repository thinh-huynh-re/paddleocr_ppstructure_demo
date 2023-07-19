# Installation

## For Ubuntu 20.x
```bash
python3.8 -m venv env
source env/bin/activate
# pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
pip install paddleclas
pip install paddleocr
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```

## For Ubuntu 22.x

### Install OpenSSL 1.1 

```bash
$ mkdir $HOME/opt && cd $HOME/opt
# Download a supported openssl version. e.g., openssl-1.1.1o.tar.gz or openssl-1.1.1t.tar.gz
$ wget https://www.openssl.org/source/openssl-1.1.1o.tar.gz
$ tar -zxvf openssl-1.1.1o.tar.gz
$ cd openssl-1.1.1o
$ ./config && make && make test
$ mkdir $HOME/opt/lib
$ mv $HOME/opt/openssl-1.1.1o/libcrypto.so.1.1 $HOME/opt/lib/
$ mv $HOME/opt/openssl-1.1.1o/libssl.so.1.1 $HOME/opt/lib/
```
