# rag-privacy
## Installing `pyserini`
You will need `pyserini` for RIC-LM.
First make sure that you've installed `torch` and `python>=3.10`. To install `pyserini`, do the following in your virtual environment:
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021
conda install -c conda-forge openjdk=21
pip install pyserini
```
Then you can check whether the installation is successful by:
```bash
python
>>> import pyserini
>>> from pyserini.search.lucene import LuceneSearcher
```
If there is no error, then the installation has no problem.
However, you might encounter one error:
```bash
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by ...)
```
Then you should do the following first every time you run the scripts:
```bash
export LD_LIBRARY_PATH=/path/to/your/conda/envs/your_env_name/lib
```
Then things should be good now.