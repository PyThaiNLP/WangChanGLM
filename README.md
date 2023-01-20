# ChomGPT
ChomGPT - ChatGPT-like model for Thai

### How to install trl

1. Clone Transformer Reinforcement Learning (trl)

> git clone https://github.com/lvwerra/trl.git

2. copy the patch to trln and apply the patch

> cp trl_patch/0001-add-stop-token-commits.patch trl/0001-add-stop-token-commits.patch

> cd trl_patch

> git apply 0001-add-stop-token-commits.patch

3. Install

> pip install -e .