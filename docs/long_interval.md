# Colabでの長区間A/B実験

20,000更新の同一チェックポイントから、A（現行）とB（非対角の25%を一様分布から抽選）をそれぞれ22,000更新まで学習する。前回のColab実験のDrive保存先を既定値とする。

## 実行

GPUランタイムで実行する。各ランタイムの初回だけ次のセルを実行する。

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone -b experiment/long-interval-ab --single-branch https://github.com/kazukitakayamas/weak_mean_flow.git /content/weak_interval_ab
%cd /content/weak_interval_ab
%pip -q install -r requirements.txt
```

```bash
%%bash
set -e
python -m unittest discover -s tests -p test_long_interval.py
python -u long_interval.py train
```

```python
!python -u long_interval.py eval
```

前回の `MyDrive/WeakMeanFlow/diagnosis_runs/weak_baseline_seed42_v1/weak_*/step_020000.pt` を優先し、なければ同じ場所の `latest.pt` を探す。候補が複数なら、train と eval に同じ `--source "/content/drive/MyDrive/.../step_020000.pt"` を加える。CIFAR-10は `MyDrive/WeakMeanFlow/data` の展開済みデータまたは公式tar.gzを使う。変更には `--data-root PATH` を使う。

分岐先は元チェックポイントの隣の `long_interval_ab/A/latest.pt` と `B/latest.pt`。コピーしたモデル、生の重みを平均した2種類のEMA、Adam、CPU/CUDAの乱数状態、テスト関数の乱数状態を用いる。再実行は各条件の最後の保存から継続し、22,000更新に達した条件は学習をスキップする。A・Bを別セッションで順に実行する場合は `train --arm A` と `train --arm B` を使う。

## 式と実装

1. **時刻の抽選** — `weakflow/time_sampling.py` の `sample_imf_times`。非対角の各組で独立に確率 $\rho$ で一様乱数を2個引き、小さい方を $r$、大きい方を $t$ とする。残りは従来のlogit-normal。対角は従来どおり。Bの追加乱数は独立した生成器を使い、seedと通算stepで再現する。Aの $\rho=0$、Bの $\rho=0.25$ は `long_interval.py` の `ARMS` に対応する。

2. **密度補正** — `off_time_score(r,t,cfg)` の戻り値を `weakflow/loss.py::weak_summary` の `aphi` に `score * phi` として加える。$p$ を一変量のlogit-normal密度とし、$0<r<t<1$ とすると、

   $$q_0(r,t)=2p(r)p(t),\qquad q_\rho(r,t)=(1-\rho)q_0(r,t)+2\rho.$$

   一様成分の密度2は一定なので、$r$ を固定して $t$ で微分すると、

   $$\partial_t\log q_\rho=\frac{(1-\rho)q_0}{(1-\rho)q_0+2\rho}\,\partial_t\log q_0.$$

   コードの `base` は従来の $\partial_t\log q_0$。係数は対数密度を使い `torch.sigmoid(log_odds)` で計算する。混合成分を選んだ結果にかかわらず、各組で混合密度のスコアを使う。この比較は弱モーメントの積分に使う時刻分布も変更する。

3. **長区間を実測** — `weakflow/trainer.py` の `h = (t-r)[len(t)//2:]`。更新ごとに $\sum_i 1[h_i\geq 0.5]$ と $\sum_i 1[h_i\geq 0.9]$ を記録する。完了後に各条件の `interval_counts.csv` に件数を保存し、1更新あたりの平均と非対角に占める割合を表示する。標準設定では非対角512組。$P_U(h\ge a)=(1-a)^2$ とlogit-normalの数値積分による期待値は次のとおり。

   | 512組あたりの期待数 | A | B |
   |---|---:|---:|
   | $h\ge0.5$ | 43.33 | 64.50 |
   | $h\ge0.9$ | 0.0074 | 1.286 |

4. **生成品質と1・16ステップの差** — `long_interval.py::evaluate_arms`。主指標は1-NFE FID。16-NFE FIDも同時に出す。既定は両条件とも `ema`（前回の図と同じ）、生成10,000枚（seed 42）、実画像はCIFAR-10学習画像50,000枚。実画像の特徴統計を共用する。別途、同じ128個のノイズ（seed 12345）で各条件の $\sqrt{\mathrm{mean}[(\hat x_1-\hat x_{16})^2]}$ をclamp前に計算する。最初の16枚は4×4で保存する。1/16ステップの一致度は生成品質と別に読む。

`long_interval_ab/eval_ema_<時刻>/` の `fid.csv`、`diagnostics.csv`、`A_1nfe.png`、`B_1nfe.png`、`A_16nfe.png`、`B_16nfe.png` が比較結果。先に画像とRMSだけ見るなら `eval --images-only`。EMAは0.9999なので、追加2,000更新後も分岐時のEMAの係数は約0.819。短期の反応は同じコマンドに `--weights raw` または `--weights ema_099` を加え、AとBで同じ重み種別を比較する。
