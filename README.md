# **CaDiff: Causal Preference Modeling via Conditional Diffusion for Multi-behavior Recommendation**

## 📚 Overview of CaDiff

We propose **CaDiff**, our main contributions are summarized as follows:

- We identify and address the limitations of existing multi-behavior recommendation methods, which overlook the confounding effect of intrinsic multi-behavior bias, proposing a novel application of causal intervention to explicitly remove this effect from auxiliary–target correlations.
- We introduce CaDiff, a framework that conditions the diffusion process on auxiliary behaviors to progressively denoise target-behavior preferences, achieving robust and unbiased multi-behavior recommendation.
- Extensive experiments on three public datasets validate the superiority, robustness, and efficiency of CaDiff.



## 📥 Data


| #Dataset | #Users | #Items | #Views    | #Collects | #Carts  | #Buys   |
| -------- | ------ | ------ | --------- | --------- | ------- | ------- |
| Taobao   | 15,449 | 11,953 | 873,954   | -         | 195,476 | 92,180  |
| Tmall    | 41,738 | 11,953 | 1,813,498 | 221,514   | 1,996   | 255,586 |
| Jdata    | 93,334 | 24,624 | 1,681,430 | 45,613    | 49,891  | 321,883 |


## 🚀 Example to Run the Codes

- Taobao dataset

```
python main.py --dataset taobao
```

- Tmall dataset

```
python main.py --dataset tmall
```

- Jdata dataset

```
python main.py --dataset jdata --alpha 0.5
```



## 📂 Folder Structure

The released code consists of the following files.

```
--data
    --taobao
    --tmall
    --jdata
--dataloader.py
--main.py
--model.py
--params.py
--trainer.py
--utils.py
```

