Use conda:

> conda env create -f environment.yml

but deepspeed in conda don't support pytorch 2.0, so you must to upgrade by pip after create environment by conda.

> conda activate name

> DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.8.3
