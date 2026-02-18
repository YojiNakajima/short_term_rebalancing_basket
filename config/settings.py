PV_LEVERAGE = 5 #Portfolio Value を有効証拠金の何倍にするか
TPV_LEVERAGE = 0.9 # PV_LEVERAGEに対し何倍の値をTargetPortfolioValueとするか

#
CIRCUIT_BREAKER_THRESHOLD = 0.5

# 部分利確のATRの基準
TP_ATR_1ST=1.0
TP_ATR_2ND=1.5
TP_ATR_3RD=1.9

# 部分利確のLot量
TP_AMOUNT_1ST = 0.5
TP_AMOUNT_2ND = 0.3
TP_AMOUNT_3RD = 0.2

# 部分利確後どの程度価格が下がったら追加ポジションを持つかのATR%の基準
REENTRY_ATR_1ST=0.8
REENTRY_ATR_2ND=1.2
REENTRY_ATR_3RD=1.6

# 追加ポジションのLot量
REENTRY_AMOUNT_1ST = 0.5
REENTRY_AMOUNT_2ND = 0.3
REENTRY_AMOUNT_3RD = 0.2

# TPV算出時に用いる実効レバレッジ（デフォルト=1.0で従来のTPV=PV*TPV_LEVERAGEと一致）
LEVERAGE = 1.0

# TPV（equity * TPV_LEVERAGE）に対して、1ATR（ATR%）変動で許容するリスク割合
# 例: 0.01 = TPVの1%をリスクにする
RISK_PCT = 0.01

ATR_PERIOD = 14
TIME_FRAME = 4 # 4=4H, 1=1H

# --- Rebalance settings ---
# If True, rebalance_portfolio will send real MT5 orders by default.
# You can still override per run via CLI flags: --trade / --dry-run.
REBALANCE_TRADE_DEFAULT = True

# --- Correlation / Risk Parity (ERC) settings ---
# EWMA half-life in bars for covariance (correlation) estimation on TIME_FRAME.
# 例: H4でhalf_life=90本 ≒ 約15営業日（1日6本換算）
CORR_EWMA_HALF_LIFE_BARS = 90

# Number of historical bars to fetch for covariance estimation.
# (log return is computed from close-to-close; an additional bar is used internally.)
CORR_LOOKBACK_BARS = 1200

# Clip absolute log returns to reduce outlier impact (e.g. news spikes).
# 例: 0.05 は単純近似で約±5%程度の変動を上限にするイメージ。
CORR_LOG_RETURN_ABS_CLIP = 0.05

# Ridge regularization strength added to covariance diagonal (scaled by avg variance).
CORR_COV_RIDGE = 1e-6

# ERC solver parameters
ERC_MAX_ITER = 5000
ERC_TOL = 1e-8
ERC_STEP = 0.5

# Target Symbols
target_symbols = {
    'XAUUSD': True,
   # 'XAGUSD': True,
   # 'XPTUSD': True,
   # 'XCUUSD': True,
    'USOIL': True,
    #'XNGUSD': True,
    'DXY': True,
    'US500': True,
    'US30': True,
    'USTEC': True,
    'STOXX50': True,
    'DE30': True,
    'FR40': True,
    'UK100': True,
    'JP225': True,
   # 'HK50': True,
    # 'AUS200': True
}
