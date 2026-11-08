# Weak Mean Flow

CIFAR-10を用いた弱形式Mean Flowの学習・画像生成・FID評価を行う。

この手順は、研究室サーバーに以下を準備済みであることを前提とする。

- リポジトリ：`~/projects/weak_mean_flow`
- Dockerイメージ：`kazuki-weak-mean-flow:torch2.5.1-cu121`
- Dockerコンテナ：`kazuki-weak-mean-flow`
- 展開済みCIFAR-10：`~/weak_mean_flow_storage/data/cifar-10-batches-py`

Dockerfileや依存ライブラリを変更しない限り、毎回のビルドは不要。
CIFAR-10も毎回ダウンロードする必要はない。

## 0. コマンドを実行する場所

このREADMEのコマンドは、原則としてサーバー側のBashで実行する。

表示例：

```text
kazuki@hsd1:~/projects/weak_mean_flow$
```

`docker exec` を使い、サーバー側からコンテナ内のPythonを実行する。
コンテナのシェルに入る必要はない。

`<RUN_DIR>` などの表記は、実際の値に置き換える。
全角スペースをコマンドに入れない。

## 1. サーバーに接続する

手元のPCから接続する。接続先は研究室の設定に合わせる。

```bash
ssh kazuki@hsd1
```

サーバーに接続したら、作業場所と保存先を設定する。

```bash
cd ~/projects/weak_mean_flow

WMF_STORAGE="$HOME/weak_mean_flow_storage"
mkdir -p "$WMF_STORAGE/outputs/logs"
```

ここで設定したシェル変数は、SSH接続を終了すると失われる。
接続し直した場合は再設定する。

## 2. コンテナとGPUの状態を確認する

```bash
docker ps -a --filter name=kazuki-weak-mean-flow
nvidia-smi
```

- `Up ...`：コンテナは起動済み。
- `Exited ...`：次のコマンドで起動する。
- コンテナが表示されない：初回作成、または削除後の再作成が必要。

停止中の場合だけ実行する。

```bash
docker start kazuki-weak-mean-flow
```

すでに学習・評価が動いていないか確認する。

```bash
docker top kazuki-weak-mean-flow
```

`python ... train.py` や `python ... evaluate.py` がある場合は、
その処理を確認してから次へ進む。同じ学習を重複して起動しない。

GPUは他の利用者と共有しているため、研究室の利用ルールに従う。

## 3. コードを更新する場合

新しい実験を始める前に、必要に応じて更新する。
実行中の学習・評価がある間は更新しない。

```bash
git status --short
git pull --ff-only
```

ローカル変更との競合などで失敗した場合は、強制的に上書きせず内容を確認する。

PythonコードやYAMLはサーバーからコンテナへマウントされているため、
変更は次に起動するPythonプロセスに反映される。

Dockerfile・requirements.txt・constraints-docker.txtを変更した場合は、
イメージの再ビルドとコンテナの再作成が必要。
既存コンテナを再起動するだけでは、新しいイメージに切り替わらない。

## 4. データとGPUを確認する

初回実行時や、環境・データの配置を変更したときに実行する。

```bash
docker exec -i \
  -w /workspace/weak_mean_flow \
  kazuki-weak-mean-flow \
  python - <<'PY'
import torch
from weakflow.data import load_cifar10

x = load_cifar10("/data/cifar10", download=False)

print("Data shape:", tuple(x.shape))
print("Data dtype:", x.dtype)
print("CUDA available:", torch.cuda.is_available())

assert torch.cuda.is_available(), "CUDA is not available"
print("GPU:", torch.cuda.get_device_name(0))
PY
```

想定される表示：

```text
Data shape: (50000, 3, 32, 32)
Data dtype: torch.uint8
CUDA available: True
GPU: Quadro RTX 8000
```

`data_batch_1 が 0 個あります` と表示された場合は、
CIFAR-10の配置とマウント先を確認する。

設定の `download: false` はそのまま使用する。
データは保存済みのものを読み込む。

## 5. 学習条件を確認する

設定ファイル：

```text
configs/weak_logitnormal_w20.yaml
```

主な設定：

| 項目 | 現在の値 |
|---|---:|
| total_steps | 20000 |
| batch_size | 1024 |
| micro_batch | 128 |
| lambda_off | 20.0 |
| ema_decay | 0.9999 |
| ema_decay_099 | 0.99 |
| log_every | 200 |
| save_every | 1000 |

実験条件を変える場合は、別名のYAMLを作り、
`experiment_name` も条件を識別できる名前にすると管理しやすい。

`--total-steps` を指定すると、YAMLの `total_steps` より優先される。

以降は、次のどちらか一方を選ぶ。

- 新しく学習する：6-A
- 保存済みモデルから再開する：6-B

## 6-A. 新しい学習を開始する

以下は、現在の設定で20,000更新する例。

ログの保存先を決める。

```bash
WMF_LOG="/outputs/weak_mean_flow/logs/train_$(date +%Y%m%d_%H%M%S).log"
echo "$WMF_LOG"
```

以下を一度だけ実行する。

```bash
docker exec -d \
  -w /workspace/weak_mean_flow \
  kazuki-weak-mean-flow \
  sh -c '
    python -u train.py \
      --config configs/weak_logitnormal_w20.yaml \
      --data-root /data/cifar10 \
      --output-root /outputs/weak_mean_flow \
      --device cuda \
      --total-steps 20000 \
      > "$1" 2>&1

    wmf_status=$?
    printf "\nexit_code=%s\n" "$wmf_status" >> "$1"
    exit "$wmf_status"
  ' sh "$WMF_LOG"
```

`-d` はバックグラウンド実行。
コマンドがすぐ終了して画面に何も表示されなくても正常。

出力とエラーはログファイルに保存される。
起動直後に7節の方法でログを確認する。

新規学習では、日時と識別子を含む新しい実験フォルダが作られる。
保存先はログ冒頭の `保存先:` に表示される。

## 6-B. 保存済みモデルから学習を再開する

再開したい実験のチェックポイントを指定する。
`<RUN_DIR>` は実際のフォルダ名に置き換える。

```bash
WMF_CKPT="/outputs/weak_mean_flow/weak_logitnormal_w20_dualema/<RUN_DIR>/latest.pt"
WMF_LOG="/outputs/weak_mean_flow/logs/resume_$(date +%Y%m%d_%H%M%S).log"
```

ファイルが存在することを確認する。

```bash
docker exec kazuki-weak-mean-flow ls -lh "$WMF_CKPT"
```

以下を一度だけ実行する。

```bash
docker exec -d \
  -w /workspace/weak_mean_flow \
  kazuki-weak-mean-flow \
  sh -c '
    python -u train.py \
      --resume "$2" \
      --data-root /data/cifar10 \
      --device cuda \
      --total-steps 20000 \
      > "$1" 2>&1

    wmf_status=$?
    printf "\nexit_code=%s\n" "$wmf_status" >> "$1"
    exit "$wmf_status"
  ' sh "$WMF_LOG" "$WMF_CKPT"
```

`--total-steps 20000` は「追加で20,000更新」ではなく、
「累計20,000更新まで」という意味。

再開時の注意：

- 設定はチェックポイントから読み込まれる。
- YAMLを編集しても、その変更は再開時の設定に自動反映されない。
- 保存先はチェックポイントと同じ実験フォルダになる。
- CPUのsmoke実験はCPUで、本番のCUDA実験はCUDAで再開する。
- 保存済み更新数が20,000以上なら、より大きい目標更新数が必要。
- 同じチェックポイントを使う学習を同時に複数起動しない。

## 7. 学習の進捗・エラーを確認する

直近のログを表示する。

```bash
docker exec kazuki-weak-mean-flow tail -n 50 "$WMF_LOG"
```

ログを継続表示する。

```bash
docker exec -it kazuki-weak-mean-flow tail -n 50 -f "$WMF_LOG"
```

この `tail -f` は、Ctrl+Cで終了してよい。
終了するのはログ表示だけで、学習は継続する。

学習プロセスとGPUの状態も確認できる。

```bash
docker top kazuki-weak-mean-flow
nvidia-smi
```

現在の設定では、ログ出力は最初の1更新目と200更新ごと。
再開時は次の200の倍数まで途中ログが出ない場合がある。

以前の実測では10更新に237秒かかった。
同じ速度なら、10更新目から200更新目まで約75分かかる。
速度はGPUの利用状況などによって変わる。

画面やログが静かなだけで、停止したとは判断しない。

### SSH接続を切る場合

バックグラウンド学習の起動とログを確認できたら、
SSH接続を終了しても学習は継続する。

学習中はコンテナを停止しない。

接続し直した場合、ログ一覧はサーバー側で確認できる。

```bash
ls -lt "$HOME/weak_mean_flow_storage/outputs/logs"
```

対象のファイル名を選び、変数を設定し直す。

```bash
WMF_LOG="/outputs/weak_mean_flow/logs/実際のログファイル名.log"
```

## 8. 学習完了を確認する

正常終了すると、ログ末尾に次の情報が表示される。

```text
今回の実行時間: ...
チェックポイント: .../latest.pt

exit_code=0
```

- `exit_code=0`：正常終了。
- 0以外：エラー終了。直前のエラーログを確認する。
- 終了コードがない：実行中、またはコンテナ停止などで終了処理まで到達していない可能性がある。

コンテナが `Up` であるだけでは、学習が動いているとは限らない。
このコンテナは学習終了後も待機状態で動き続ける。

チェックポイントは通常1,000更新ごとと学習終了時に保存される。
設定された節目では `step_002000.pt` なども保存される。

## 9. 画像生成・FID評価

学習が正常終了した後、ログに表示されたチェックポイントを指定する。

```bash
WMF_CKPT="/outputs/weak_mean_flow/weak_logitnormal_w20_dualema/<RUN_DIR>/latest.pt"
WMF_LOG="/outputs/weak_mean_flow/logs/eval_$(date +%Y%m%d_%H%M%S).log"
```

評価もバックグラウンドで実行する。

```bash
docker exec -d \
  -w /workspace/weak_mean_flow \
  kazuki-weak-mean-flow \
  sh -c '
    python -u evaluate.py \
      --checkpoint "$2" \
      --data-root /data/cifar10 \
      --weights raw ema_099 ema \
      --nfe 1 4 16 \
      --samples 10000 \
      --batch-size 64 \
      --seed 42 \
      > "$1" 2>&1

    wmf_status=$?
    printf "\nexit_code=%s\n" "$wmf_status" >> "$1"
    exit "$wmf_status"
  ' sh "$WMF_LOG" "$WMF_CKPT"
```

これは、3種類の重み × 3種類のNFEの計9条件を評価する。

| 指定 | 使用する重み |
|---|---|
| raw | 学習モデル本体 |
| ema_099 | EMA decay 0.99 |
| ema | EMA decay 0.9999 |

進捗確認：

```bash
docker exec -it kazuki-weak-mean-flow tail -n 50 -f "$WMF_LOG"
```

正常終了すると `評価結果:` と `exit_code=0` が表示される。

評価結果は、チェックポイントがあるフォルダ内の
`evaluation/` 以下に保存される。

主な出力：

- 4×4の生成画像：PNG
- FID評価値：`fid.csv`
- 評価条件：`settings.json`

FID用のモデル重みは、初回にダウンロードが発生することがある。
`/cache` のマウントを維持して再利用する。

## 10. 結果の保存場所

| 内容 | サーバー側 | コンテナ側 |
|---|---|---|
| ソースコード | `~/projects/weak_mean_flow` | `/workspace/weak_mean_flow` |
| CIFAR-10 | `~/weak_mean_flow_storage/data` | `/data/cifar10` |
| 学習・評価結果 | `~/weak_mean_flow_storage/outputs` | `/outputs/weak_mean_flow` |
| キャッシュ | `~/weak_mean_flow_storage/cache` | `/cache` |

マウント設定の確認：

```bash
docker inspect kazuki-weak-mean-flow \
  --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

データ・結果・キャッシュの実体はサーバー側にある。
コンテナを削除しても、これらのフォルダは残る。

ただし、サーバー側のフォルダを直接削除するとデータは失われる。
重要な実験結果は別の保存先にもバックアップする。

## 11. 作業終了後の後片付け

### 通常はコンテナを停止するだけ

学習・評価の正常終了をログで確認し、
プロセス一覧に実行中の処理がないことを確認する。

```bash
docker top kazuki-weak-mean-flow
```

終了済みなら、自分のコンテナを停止する。

```bash
docker stop kazuki-weak-mean-flow
```

停止を確認する。

```bash
docker ps -a --filter name=kazuki-weak-mean-flow
```

次回は、以下で再利用できる。

```bash
docker start kazuki-weak-mean-flow
```

コンテナの起動だけでは、学習は自動再開しない。
途中から再開する場合は6-Bを実行する。

### 途中で停止する場合

現在の学習コードは、停止要求を受けた時点で
自動的にチェックポイントを保存する仕組みにはなっていない。

コンテナを停止すると学習も止まり、
最後に保存されたチェックポイント以降の更新は失われる。
再開は6-Bの手順で行う。

### コンテナを削除する場合だけ

不要になった停止済みの自分のコンテナを削除する場合：

```bash
docker rm kazuki-weak-mean-flow
```

削除後は `docker start` では起動できないため、再作成が必要。

Dockerfile・イメージ・CIFAR-10を毎回削除する必要はない。
共有サーバーでは、他の利用者のコンテナを停止・削除しない。
`docker system prune` などの一括削除は日常の後片付けに使用しない。

## 12. 削除したコンテナを再作成する場合

この節は、コンテナを削除した場合にだけ実行する。

```bash
cd ~/projects/weak_mean_flow

WMF_STORAGE="$HOME/weak_mean_flow_storage"
WMF_GPU_ID=0

mkdir -p "$WMF_STORAGE/data"
mkdir -p "$WMF_STORAGE/outputs/logs"
mkdir -p "$WMF_STORAGE/cache"

docker run -d \
  --name kazuki-weak-mean-flow \
  --gpus "device=${WMF_GPU_ID}" \
  --user "$(id -u):$(id -g)" \
  --shm-size=2g \
  --mount "type=bind,source=$PWD,target=/workspace/weak_mean_flow" \
  --mount "type=bind,source=$WMF_STORAGE/data,target=/data/cifar10,readonly" \
  --mount "type=bind,source=$WMF_STORAGE/outputs,target=/outputs/weak_mean_flow" \
  --mount "type=bind,source=$WMF_STORAGE/cache,target=/cache" \
  kazuki-weak-mean-flow:torch2.5.1-cu121
```

同じ保存先をマウントするので、CIFAR-10や学習結果を引き継げる。
データの再ダウンロードは不要。

イメージがない場合や依存関係を更新した場合は、
コンテナ作成前にビルドする。

```bash
docker build -t kazuki-weak-mean-flow:torch2.5.1-cu121 .
```

## 13. よくある表示・エラー

| 表示・状況 | 意味と対応 |
|---|---|
| `docker exec -d` 後に何も表示されない | バックグラウンド実行の通常動作。ログを確認する |
| ログが長時間増えない | 200更新ごとの出力設定を確認。プロセスとGPUの状態も見る |
| `--device: expected one argument` | `--device cpu` または `--device cuda` と値まで指定する |
| `data_batch_1 が 0 個あります` | データの配置とマウント先を確認する |
| `I have no name!` | 数値のユーザーIDに対応する名前がコンテナ内にない。これだけでは学習エラーではない |
| `No such container` | コンテナ名を確認。削除済みなら12節で再作成する |
| `container ... is not running` | 停止中のコンテナを `docker start` で起動する |
| コンテナはUpなのに学習していない | 学習終了後も待機する構成。ログと `docker top` で確認する |
