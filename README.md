# weak_mean_flow

### ステップ1: 追加依存を入れる
```python
python -m pip install -r requirements.txt
```

### ステップ2-1: ダミーデータで動作確認
```python
python train.py \
  --config configs/weak_logitnormal_w20.yaml \
  --smoke \
  --device cpu \
  --total-steps 3 \
  --output-root ./outputs
```
### ステップ2-2: 2-1の処理後に表示された`latest.pt`を指定し、5更新まで再開
```python
python train.py \
  --resume "./outputs/weak_logitnormal_w20_dualema_smoke/<RUN_DIR>/latest.pt" \
  --total-steps 5 \
  --device cpu
```

### ステップ3: 本番学習
<br>
.yamlの`total_steps: 20000`を使用

```python
python train.py \
  --config configs/weak_logitnormal_w20.yaml \
  --data-root /data/cifar10 \
  --output-root /outputs/weak_mean_flow \
  --device cuda
　--total-steps 10
```

### ステップ4: 画像生成とFID評価
<br>
Dockerでは、`/data/cifar10`と`/outputs/weak_mean_flow`を、それぞれサーバー上のデータ領域・保存領域へマウントが必要

```python
python evaluate.py \
  --checkpoint "/outputs/weak_mean_flow/weak_logitnormal_w20_dualema/<RUN_DIR>/latest.pt" \
  --data-root /data/cifar10 \
  --weights raw ema_099 ema \
  --nfe 1 4 16 \
  --samples 10000 \
  --batch-size 64 \
  --seed 42
```
