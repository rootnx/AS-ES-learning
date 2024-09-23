# AS-ES-learning
Implementation for ACL2024 paper "[AS-ES Learning: Towards Efficient CoT Learning in Small Models](https://aclanthology.org/2024.findings-acl.635/)"

## Datasets
We use the following datasets in our paper:
* MWP - [Distilling Chain-of-Thought Reasoning from code-davinci-002 to FlanT5](https://github.com/FranxYao/FlanT5-CoT-Specialization): the in-context chain-of-thought part of the data
* PET report summary - cPET-11K, a novel dataset we release in this paper (under `/data`). A collection of PET/CT report data from patients with pancreatic cancer.

## Quick Start
* Environment configurations: `conda env create -f environment.yml`
* Usage Example:
  * for as-es dataset construction, run `bash utils/run.sh`
  * for training using as-es dataset, run `bash script/train.sh`
  * for inference, run `bash script/test.sh`  

## Citing
If you find our work helpful, feel free to cite our publication -

AS-ES Learning: Towards Efficient CoT Learning in Small Models

```
@inproceedings{xi-etal-2024-es,
    title = "{AS}-{ES} Learning: Towards efficient {C}o{T} learning in small models",
    author = "Xi, Nuwa  and
      Chen, Yuhan  and
      Zhao, Sendong  and
      Wang, Haochun  and
      GongZhang, GongZhang  and
      Qin, Bing  and
      Liu, Ting",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.635",
    pages = "10686--10697"
}
```
If you have any questions, feel free to contact: [Nova X](mailto:nwxi@ir.hit.edu.cn)
