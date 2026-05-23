import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from rtdl_revisiting_models import FTTransformer

warnings.filterwarnings("ignore")


# !!!!!!!!!!!!! 하이퍼파라미터 세팅값들  !!!!!!!!!!!!!!!!!
#  일반 세팅값(시드/학습장치)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# FT-Transformer 학습 파라미터
N_EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 15

# FT-Transformer 모델 파라미터
# D_BLOCK은 반드시 ATTENTION_N_HEADS로 나누어 떨어져야 함 (예: 192÷8=24 OK)
N_BLOCKS = 3
D_BLOCK = 192
ATTENTION_N_HEADS = 8
ATTENTION_DROPOUT = 0.2
FFN_DROPOUT = 0.1
RESIDUAL_DROPOUT = 0.0
FFN_D_HIDDEN_MULTIPLIER = 4 / 3

# 전처리 설정
USE_LOG_TARGET = True

#데이터 경로 및 컬럼 설정
#TODO: 입력 데이터명이나 경로가 바뀌면 해당 경로로 수정
DATA_PATH = "preprocessed_apt_trade_final.csv"  # 연도별 CSV가 들어있는 폴더 경로

# 랜덤 분할 비율 설정
TEST_SIZE  = 0.2   # 전체 대비 테스트 비율
VALID_SIZE = 0.25  # 나머지(train+valid) 대비 valid 비율 → 전체의 20%

#TODO: 만약 예측할 대상이 바뀌면 컬럼도 같이 변경
TARGET_COL = "price_manwon"

# 제거할 ID성/누수성 컬럼
#TODO: 전처리된 데이터에서 삭제 필요하면 적기
DROP_COLS = [
    "ID", "year_month", "log_price_manwon" , "pice_manwon"
]



# 함수 이름 : set_korean_font()
# 기능     : 결과 사진에서 한글이 깨지지 않도록 시스템에 설치된 한글 폰트를 탐색하여 적용한다.
# 파라미터 : 없음
# 반환값   : 없음
def set_korean_font() -> None:
    import matplotlib.font_manager as fm
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False


# 함수 이름 : set_seed()
# 기능     : numpy, PyTorch, CUDA의 랜덤 시드를 고정하여 실험 결과의 재현성을 보장한다.
#         : 이 기능이 있어야 반복해서 수행해도 같은 값이 나온다고함.
# 파라미터 : int seed -> 고정할 랜덤 시드 값
# 반환값   : 없음
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""
# 함수 이름 : load_and_preprocess()
# 기능     : 폴더 내 연도별 CSV 파일을 모두 로드하여 합친 후, 취소 거래 제거, 아파트 필터링,
#            이상치 제거, 불필요 컬럼 삭제, 날짜 컬럼 분해, 코드성 컬럼 범주형 변환을 수행한다.
#            TODO: 전처리 데이터 받으면 이 함수는 날릴 예정입니다.
# 파라미터 : str folder -> 연도별 CSV 파일이 들어있는 폴더 경로
# 반환값   : pd.DataFrame -> 전처리가 완료된 데이터프레임
def load_and_preprocess(folder: str) -> pd.DataFrame:
    dfs = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".csv"):
            path = os.path.join(folder, fname)
            try:
                dfs.append(pd.read_csv(path, encoding="utf-8-sig", low_memory=False))
            except UnicodeDecodeError:
                dfs.append(pd.read_csv(path, encoding="cp949", low_memory=False))
    df = pd.concat(dfs, ignore_index=True)

    print(f"\n[INFO] 데이터 크기: {df.shape}")

    #도메인 특수 처리 - 해당 컬럼이 없으면 자동으로 건너뜀
    if "취소일" in df.columns:
        df = df[df["취소일"].isnull()]

    if "건물용도" in df.columns:
        df = df[df["건물용도"].astype(str).str.contains("아파트", na=False)]

    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0]

    # 가격 이상치 범위 설정(서울시 아파트 기준: 1000만~100억)
    before = len(df)
    df = df[(df[TARGET_COL] >= 1000) & (df[TARGET_COL] <= 1000000)]
    print(f"[INFO] 가격 이상치 제거: {before:,} → {len(df):,}")

    #면적 컬럼명/범위
    if "건물면적(㎡)" in df.columns:
        before = len(df)
        df = df[(df["건물면적(㎡)"] > 10) & (df["건물면적(㎡)"] < 500)]
        print(f"[INFO] 면적 이상치 제거: {before:,} → {len(df):,}")

    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 날짜 컬럼 분해 (YYYYMMDD 형식)
    if "계약일" in df.columns:
        df["계약일"] = df["계약일"].astype(str)
        df["계약년"] = df["계약일"].str[:4].astype(int)
        df["계약월"] = df["계약일"].str[4:6].astype(int)
        df["계약일자"] = df["계약일"].str[6:8].astype(int)
        df = df.drop(columns=["계약일"])

    # 숫자지만 범주형으로 처리할 컬럼 (코드성 컬럼)
    for col in ["자치구코드", "법정동코드", "지번구분"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(str)

    return df
"""

# 함수 이름 : encode_features()
# 기능     : 수치형/범주형 컬럼을 자동 분류하고, 결측치 처리 및 LabelEncoder 인코딩을 수행한다.
#            D_BLOCK과 ATTENTION_N_HEADS의 호환성 검증 및 feature 수 관련 경고도 출력한다.
# 파라미터 : pd.DataFrame df -> 전처리가 완료된 데이터프레임
# 반환값   : pd.DataFrame df          -> 인코딩이 적용된 데이터프레임
#            list numeric_cols        -> 수치형 컬럼명 리스트
#            list categorical_cols    -> 범주형 컬럼명 리스트
#            list cat_cardinalities   -> 각 범주형 컬럼의 고유값 개수 리스트
def encode_features(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)

    print(f"[INFO] 수치형 변수 ({len(numeric_cols)}개): {numeric_cols}")
    print(f"[INFO] 범주형 변수 ({len(categorical_cols)}개): {categorical_cols}")

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cardinalities = []
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown").astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        cat_cardinalities.append(len(le.classes_))

    print(f"[INFO] 범주형 카디널리티: {cat_cardinalities}")

    n_total = len(numeric_cols) + len(categorical_cols)
    assert D_BLOCK % ATTENTION_N_HEADS == 0, \
        f"D_BLOCK({D_BLOCK})은 ATTENTION_N_HEADS({ATTENTION_N_HEADS})로 나누어 떨어져야 합니다"
    print(f"[INFO] 총 feature 수: {n_total} (수치형 {len(numeric_cols)}, 범주형 {len(categorical_cols)})")
    if n_total > 50:
        print(f"[권장] feature 수가 많습니다. D_BLOCK을 256 이상으로 올려보세요 (현재: {D_BLOCK})")
    if max(cat_cardinalities, default=0) > 1000:
        print(f"[경고] 고카디널리티 범주형 변수 존재 (최대: {max(cat_cardinalities)}). 메모리 부족 시 BATCH_SIZE 감소 권장")

    return df, numeric_cols, categorical_cols, cat_cardinalities


# 함수 이름 : split_and_scale()
# 기능     : 시간 순서대로 데이터를 train/valid/test로 분할하고, 수치형 변수에
#            QuantileTransformer를 적용하며, 타겟값에 로그 변환 및 표준화를 수행한다.
#            데이터가 year_month 기준으로 정렬되어 있어야 하며, 미래 데이터 누수를 방지한다.
#            분할 비율: TEST_SIZE(test), VALID_SIZE(train_temp 대비 valid)
# 파라미터 : np.ndarray X_numeric    -> 수치형 feature 배열 (float32)
#            np.ndarray X_categorical -> 범주형 feature 배열 (int64)
#            np.ndarray y             -> 타겟값 배열 (float32)
# 반환값   : tuple -> 분할/변환된 X_num, X_cat (train/valid/test),
#                     표준화된 y_train_s, y_valid_s, 원본 y_test,
#                     타겟 복원용 y_mean, y_std, 테스트 인덱스 idx_test
def split_and_scale(X_numeric, X_categorical, y):
    n = len(y)
    n_test     = int(n * TEST_SIZE)
    n_trainval = n - n_test
    n_valid    = int(n_trainval * VALID_SIZE)
    n_train    = n_trainval - n_valid

    idx_train = np.arange(0, n_train)
    idx_valid = np.arange(n_train, n_train + n_valid)
    idx_test  = np.arange(n_train + n_valid, n)

    X_num_train, X_num_valid, X_num_test = X_numeric[idx_train], X_numeric[idx_valid], X_numeric[idx_test]
    X_cat_train, X_cat_valid, X_cat_test = X_categorical[idx_train], X_categorical[idx_valid], X_categorical[idx_test]
    y_train, y_valid, y_test = y[idx_train], y[idx_valid], y[idx_test]

    print(f"\n[INFO] 시간순 분할 — Train: {len(idx_train)}, Valid: {len(idx_valid)}, Test: {len(idx_test)}")

    qt = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(X_num_train)),
        random_state=SEED,
    )
    X_num_train = qt.fit_transform(X_num_train).astype(np.float32)
    X_num_valid = qt.transform(X_num_valid).astype(np.float32)
    X_num_test  = qt.transform(X_num_test).astype(np.float32)

    if USE_LOG_TARGET:
        y_train_t = np.log1p(y_train)
        y_valid_t = np.log1p(y_valid)
    else:
        y_train_t = y_train.copy()
        y_valid_t = y_valid.copy()

    y_mean, y_std = y_train_t.mean(), y_train_t.std()
    y_train_s = (y_train_t - y_mean) / y_std
    y_valid_s = (y_valid_t - y_mean) / y_std

    return (
        X_num_train, X_num_valid, X_num_test,
        X_cat_train, X_cat_valid, X_cat_test,
        y_train_s, y_valid_s, y_test,
        y_mean, y_std, idx_test,
    )


# 함수 이름 : build_dataloaders()
# 기능     : train/valid/test 배열을 TensorDataset으로 묶어 PyTorch DataLoader 3개를 생성한다.
#            train은 shuffle=True, valid/test는 shuffle=False로 설정된다.
# 파라미터 : np.ndarray X_num_train  -> 학습 수치형 feature
#            np.ndarray X_num_valid  -> 검증 수치형 feature
#            np.ndarray X_num_test   -> 테스트 수치형 feature
#            np.ndarray X_cat_train  -> 학습 범주형 feature
#            np.ndarray X_cat_valid  -> 검증 범주형 feature
#            np.ndarray X_cat_test   -> 테스트 범주형 feature
#            np.ndarray y_train_s    -> 표준화된 학습 타겟
#            np.ndarray y_valid_s    -> 표준화된 검증 타겟
#            np.ndarray idx_test     -> 테스트 인덱스 (크기 참조용)
# 반환값   : tuple(DataLoader, DataLoader, DataLoader) -> train/valid/test DataLoader
def build_dataloaders(X_num_train, X_num_valid, X_num_test,
                      X_cat_train, X_cat_valid, X_cat_test,
                      y_train_s, y_valid_s, idx_test):
    def _make(X_num, X_cat, y, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X_num),
            torch.from_numpy(X_cat),
            torch.from_numpy(y).float(),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    train_loader = _make(X_num_train, X_cat_train, y_train_s, shuffle=True)
    valid_loader = _make(X_num_valid, X_cat_valid, y_valid_s, shuffle=False)
    test_loader  = _make(X_num_test,  X_cat_test,
                         np.zeros(len(idx_test), dtype=np.float32), shuffle=False)
    return train_loader, valid_loader, test_loader


# 함수 이름 : build_model()
# 기능     : FTTransformer 모델을 생성하고 AdamW optimizer와 MSE loss를 함께 반환한다.
#            모델 구조는 상단 하이퍼파라미터 설정값을 따른다.
# 파라미터 : int  n_cont_features   -> 수치형 feature 개수
#            list cat_cardinalities -> 각 범주형 feature의 고유값 개수 리스트
# 반환값   : FTTransformer model    -> 생성된 FT-Transformer 모델
#            AdamW optimizer        -> AdamW 옵티마이저
#            MSELoss criterion      -> MSE 손실 함수
def build_model(n_cont_features: int, cat_cardinalities: list):
    model = FTTransformer(
        n_cont_features=n_cont_features,
        cat_cardinalities=cat_cardinalities,
        d_out=1,
        n_blocks=N_BLOCKS,
        d_block=D_BLOCK,
        attention_n_heads=ATTENTION_N_HEADS,
        attention_dropout=ATTENTION_DROPOUT,
        ffn_d_hidden=None,
        ffn_d_hidden_multiplier=FFN_D_HIDDEN_MULTIPLIER,
        ffn_dropout=FFN_DROPOUT,
        residual_dropout=RESIDUAL_DROPOUT,
    ).to(DEVICE)

    print(f"\n[INFO] 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


# 함수 이름 : train()
# 기능     : FT-Transformer 모델을 학습한다. 매 epoch마다 train/validation loss를 계산하고,
#            early stopping 조건 충족 시 학습을 조기 종료하며 best model 가중치를 복원한다.
# 파라미터 : FTTransformer model      -> 학습할 모델
#            DataLoader train_loader  -> 학습 DataLoader
#            DataLoader valid_loader  -> 검증 DataLoader
#            Optimizer optimizer      -> AdamW 옵티마이저
#            Loss criterion           -> MSE 손실 함수
# 반환값   : dict history   -> epoch별 train_loss, valid_loss, valid_rmse 기록
#            int  best_epoch -> valid_loss가 가장 낮았던 epoch 번호
def train(model, train_loader, valid_loader, optimizer, criterion):
    print("\n[INFO] FT-Transformer 학습 시작...")
    print("=" * 70)

    best_valid_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    history = {"train_loss": [], "valid_loss": [], "valid_rmse": []}

    for epoch in range(N_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for x_num, x_cat, y_batch in train_loader:
            x_num, x_cat, y_batch = x_num.to(DEVICE), x_cat.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_num, x_cat).squeeze(-1)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y_batch)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss = 0
        valid_preds, valid_targets = [], []
        with torch.no_grad():
            for x_num, x_cat, y_batch in valid_loader:
                x_num, x_cat, y_batch = x_num.to(DEVICE), x_cat.to(DEVICE), y_batch.to(DEVICE)
                pred = model(x_num, x_cat).squeeze(-1)
                valid_loss += criterion(pred, y_batch).item() * len(y_batch)
                valid_preds.append(pred.cpu().numpy())
                valid_targets.append(y_batch.cpu().numpy())
        valid_loss /= len(valid_loader.dataset)
        valid_rmse = np.sqrt(mean_squared_error(
            np.concatenate(valid_targets), np.concatenate(valid_preds)
        ))

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_rmse"].append(valid_rmse)

        print(f"Epoch {epoch+1:3d} | train_loss: {train_loss:.4f} | valid_loss: {valid_loss:.4f} | valid_rmse: {valid_rmse:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n[INFO] Early stopping at epoch {epoch+1} (best epoch: {best_epoch})")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"\n[INFO] Best epoch: {best_epoch}, Best valid loss: {best_valid_loss:.4f}")

    return history, best_epoch


# 함수 이름 : evaluate_and_save()
# 기능     : 테스트셋에 대해 예측을 수행하고 성능 지표를 출력한다.
#            학습 곡선 및 예측 산점도를 PNG로 저장하고, 예측 결과를 CSV로 저장하며,
#            학습된 모델을 .pt 파일로 저장한다.
# 파라미터 : FTTransformer model   -> 학습이 완료된 모델
#            DataLoader test_loader -> 테스트 DataLoader
#            pd.DataFrame df        -> 전처리된 전체 데이터프레임 (결과 CSV 생성용)
#            np.ndarray idx_test    -> 테스트셋 인덱스
#            np.ndarray y_test      -> 테스트셋 실제 타겟값 (원본 스케일)
#            float y_mean           -> 타겟 표준화에 사용된 평균값
#            float y_std            -> 타겟 표준화에 사용된 표준편차
#            dict history           -> 학습 epoch별 loss 기록
#            int  best_epoch        -> best valid_loss 달성 epoch
# 반환값   : 없음
def evaluate_and_save(model, test_loader, df, idx_test, y_test, y_mean, y_std, history, best_epoch):
    # 예측
    model.eval()
    test_preds_s = []
    with torch.no_grad():
        for x_num, x_cat, _ in test_loader:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            test_preds_s.append(model(x_num, x_cat).squeeze(-1).cpu().numpy())
    test_preds_s = np.concatenate(test_preds_s)

    # 역변환
    test_preds_t = test_preds_s * y_std + y_mean
    y_pred = np.expm1(test_preds_t) if USE_LOG_TARGET else test_preds_t
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    # 지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2   = r2_score(y_test, y_pred)

    abs_pct_error = np.abs((y_test - y_pred) / y_test) * 100
    acc_within_5  = (abs_pct_error <= 5).mean() * 100
    acc_within_10 = (abs_pct_error <= 10).mean() * 100
    acc_within_20 = (abs_pct_error <= 20).mean() * 100

    print("\n" + "=" * 70)
    print("정확도 테스트 결과 (Test Set)")
    print("=" * 70)
    print(f"\n[RMSE]  {rmse:>15,.2f}  (예측값과 실제값의 평균 제곱근 오차)")
    print(f"[MAE]   {mae:>15,.2f}  (평균 절대 오차)")
    print(f"[MAPE]  {mape:>15.4f}% (평균 절대 백분율 오차)")
    print(f"[R²]    {r2:>15.4f}  (결정계수)")
    print("\n" + "=" * 70)
    print("예측 정확도 (Percent Accuracy)")
    print("=" * 70)
    print(f"[정확도-1] 100 - MAPE         : {100 - mape:>6.2f}%")
    print(f"[정확도-2] R² × 100 (설명력)   : {r2 * 100:>6.2f}%")
    print(f"[정확도-3] 허용 오차 범위 내   :")
    print(f"          오차범위 ±5%  이내 정확도   : {acc_within_5:>6.2f}%")
    print(f"          오차범위 ±10% 이내 정확도   : {acc_within_10:>6.2f}%")
    print(f"          오차범위 ±20% 이내 정확도   : {acc_within_20:>6.2f}%")
    #TODO: 나중에 저번에 본 논문처럼 결과 간략화하기

    # 예측 결과 샘플
    print("\n" + "=" * 70)
    print("예측 결과 샘플 (상위 10개)")
    print("=" * 70)
    sample_df = pd.DataFrame({
        "실제값": y_test[:10],
        "예측값": y_pred[:10],
        "오차": (y_test - y_pred)[:10],
        "오차율(%)": ((y_test - y_pred) / y_test * 100)[:10],
    })
    print(sample_df.to_string(index=False))

    # 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["valid_loss"], label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (standardized)")
    plt.title("FT-Transformer Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ft_transformer_loss_curve.png", dpi=120)
    print("\n[INFO] 학습 곡선 저장: ft_transformer_loss_curve.png")

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10)
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="이상적 예측선")
    plt.xlabel("실제 거래가")
    plt.ylabel("예측 거래가")
    plt.title(f"FT-Transformer 예측 결과 (R² = {r2:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ft_transformer_prediction_scatter.png", dpi=120)
    print("[INFO] 예측 산점도 저장: ft_transformer_prediction_scatter.png")

    # 실제값 기준 정렬 후 예측값 비교 그래프
    sort_idx = np.argsort(y_test)
    y_test_s, y_pred_s = y_test[sort_idx], y_pred[sort_idx]

    # 0~1 정규화 (그래프 스케일 통일)
    y_min, y_max = y_test_s.min(), y_test_s.max()
    y_test_norm = (y_test_s - y_min) / (y_max - y_min)
    y_pred_norm = (y_pred_s - y_min) / (y_max - y_min)

    plt.figure(figsize=(12, 5))
    plt.scatter(range(len(y_test_norm)), y_pred_norm, color="red", marker="x", s=10, alpha=0.5, label="prediction")
    plt.plot(y_test_norm, color="black", linewidth=1.5, label="actual")
    plt.xlabel("Sample Index (실제값 기준 정렬)")
    plt.ylabel("Normalized Price")
    plt.title("FT-Transformer Prediction vs Actual (Sorted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ft_transformer_sorted_prediction.png", dpi=120)
    print("[INFO] 정렬 비교 그래프 저장: ft_transformer_sorted_prediction.png")

    # CSV 저장
    results_df = df.iloc[idx_test].copy().reset_index(drop=True)
    results_df["실제_거래가(만원)"] = y_test
    results_df["예측_거래가(만원)"] = y_pred.round(2)
    results_df["오차(만원)"] = (y_test - y_pred).round(2)
    results_df["오차율(%)"] = ((y_test - y_pred) / y_test * 100).round(4)
    results_df["절대오차율(%)"] = abs_pct_error.round(4)

    results_df.to_csv("ft_transformer_predictions.csv", index=False, encoding="utf-8-sig")
    results_df.sort_values("절대오차율(%)", ascending=False).to_csv(
        "ft_transformer_predictions_by_error.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n[INFO] 예측 결과 저장: ft_transformer_predictions.csv / ft_transformer_predictions_by_error.csv")
    print(f"       총 {len(results_df):,}개 예측 결과 저장")

    # 모델 저장
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "test_metrics": {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2},
    }, "ft_transformer_apartment_model.pt")
    print("\n[INFO] 학습된 모델 저장: ft_transformer_apartment_model.pt")


# 함수 이름 : main()
# 기능     : 전체 파이프라인을 순서대로 실행한다.
#            폰트/시드 설정 → 데이터 로드 및 전처리 → 인코딩 → 분할/스케일링
#            → DataLoader 생성 → 모델 생성 → 학습 → 평가 및 저장
# 파라미터 : 없음
# 반환값   : 없음
def main():
    set_korean_font()
    set_seed(SEED)
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] PyTorch version: {torch.__version__}")

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig") #TODO: 전처리 함수 날리면 df = pd.read_csv(DATA_PATH, encoding="utf-8-sig") 이걸로 교체하기

    # 시간순 정렬 (미래 데이터 누수 방지 — split_and_scale이 순차 분할을 가정)
    if "year_month" in df.columns:
        df = df.sort_values("year_month").reset_index(drop=True)
        print(f"[INFO] year_month 범위: {df['year_month'].min()} ~ {df['year_month'].max()}")

    # 불필요 컬럼 제거
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    df, numeric_cols, categorical_cols, cat_cardinalities = encode_features(df)

    X_numeric     = df[numeric_cols].values.astype(np.float32)
    X_categorical = df[categorical_cols].values.astype(np.int64)
    y             = df[TARGET_COL].values.astype(np.float32)

    (X_num_train, X_num_valid, X_num_test,
     X_cat_train, X_cat_valid, X_cat_test,
     y_train_s, y_valid_s, y_test,
     y_mean, y_std, idx_test) = split_and_scale(X_numeric, X_categorical, y)

    train_loader, valid_loader, test_loader = build_dataloaders(
        X_num_train, X_num_valid, X_num_test,
        X_cat_train, X_cat_valid, X_cat_test,
        y_train_s, y_valid_s, idx_test,
    )

    model, optimizer, criterion = build_model(len(numeric_cols), cat_cardinalities)

    history, best_epoch = train(model, train_loader, valid_loader, optimizer, criterion)

    evaluate_and_save(model, test_loader, df, idx_test, y_test, y_mean, y_std, history, best_epoch)

    print("\nFT-Transformer 학습 및 평가 완료!")


if __name__ == "__main__":
    main()
