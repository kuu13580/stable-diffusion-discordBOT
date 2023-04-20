# 必要なライブラリ
- diffusers
- discord
- torch類([pytorchサイト](https://pytorch.org/)からpip)
- transformers
- accelerate
- omegaconf

# 各種変換
## ckpt -> diffusers

```
python .\convert_original_stable_diffusion_to_diffusers.py --checkpoint_path "./path/to/ckpt" --dump_path "./path/to/models/<output>"
```
## safetensors -> ckpt
[ツール](https://github.com/diStyApps/Safe-and-Stable-Ckpt2Safetensors-Conversion-Tool-GUI)を使わせてもらう
## VAEの入れ替え
[スクリプト](https://note.com/kohya_ss/n/nf5893a2e719c)を使わせてもらう
```
python merge_vae.py <挿入先ckpt> <入れ替えるVAE> <出力先>
```
# nsfw無効化
stable-diffusion-v2はプログラム内で無効化できないので
diffusersを改変する
```
$ python
>>> import diffusers
>>> print(diffusers.__file__)
```
でフォルダの場所を見つけて、`diffusers/pipelines/stable_diffusion/safety_check.py`内の記述をコメントアウト