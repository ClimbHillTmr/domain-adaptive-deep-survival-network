import argparse
import os
import sys
import faulthandler
faulthandler.enable()
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except Exception:
    pass

# 兼容脚本运行的模块路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import os
import sys

# 兼容脚本运行的模块路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.loader import load_and_build_features, load_seq_static_survival
from src.training.runner import run_once, run_train_external


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="输入CSV路径（兼容旧用法，等同于 --train_csv）")
    parser.add_argument(
        "--train_csv",
        default="data_preprocessing/data/深医_final_data.csv",
        help="训练/内部验证CSV路径",
    )
    parser.add_argument(
        "--external_csv",
        default="data_preprocessing/data/福鼎_final_data.csv",
        help="外部验证CSV路径",
    )
    parser.add_argument("--save_dir", default="./runs", help="输出目录")
    parser.add_argument("--b1", type=float, default=30.0, help="早期边界(min)")
    parser.add_argument("--b2", type=float, default=60.0, help="中期边界(min)")
    parser.add_argument("--b3", type=float, default=120.0, help="晚期边界(min)")
    parser.add_argument(
        "--boundaries_mode",
        choices=["absolute", "relative"],
        default="absolute",
        help="边界类型：absolute按分钟；relative为会话时长比例(0-1)",
    )
    parser.add_argument(
        "--model",
        choices=["lgbm", "transformer"],
        default=["lgbm"],
        help="模型类型",
    )
    parser.add_argument(
        "--survival", choices=["on", "off"], default="off", help="是否启用生存辅助"
    )
    parser.add_argument(
        "--config",
        default="configs/defaults.yaml",
        help="配置文件路径（列映射/边界/阈值）",
    )
    parser.add_argument(
        "--auto_boundaries",
        choices=["on", "off"],
        default="on",
        help="自动统计加权分位数边界",
    )
    parser.add_argument(
        "--calibration",
        choices=["sigmoid", "isotonic", "off"],
        default="sigmoid",
        help="概率校准方法：sigmoid/isotonic/off",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="外部概率温度缩放系数(>0)，1为不缩放",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=0.8,
        help="外部分布再加权剪裁下限",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=1.5,
        help="外部分布再加权剪裁上限",
    )
    parser.add_argument(
        "--coral",
        choices=["on", "off"],
        default="on",
        help="是否对外部特征执行CORAL对齐",
    )
    parser.add_argument(
        "--scaler",
        choices=["robust", "quantile"],
        default="robust",
        help="特征缩放器：robust 或 quantile",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=3,
        help="LightGBM 交叉验证折数",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,17,99",
        help="集成随机种子，逗号分隔",
    )
    parser.add_argument(
        "--ext_min_thresholds",
        type=str,
        help="外部最小阈值列表，逗号分隔，例如 0.12,0.10,0.12,0.12",
    )
    parser.add_argument(
        "--floor_eps",
        type=str,
        help="外部概率下限向量，逗号分隔",
    )
    parser.add_argument(
        "--mix_alpha",
        type=float,
        default=0.0,
        help="外部概率与先验混合系数(0-0.5)",
    )
    args = parser.parse_args()

    train_csv = args.train_csv or args.csv
    import yaml

    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    boundaries = (args.b1, args.b2, args.b3)
    # 自动加权分位数边界（基于最终CSV的 events/et_min）
    if args.auto_boundaries == "on" and args.train_csv and args.external_csv:
        import pandas as pd, numpy as np, time

        df_tr = pd.read_csv(args.train_csv)
        df_ex = pd.read_csv(args.external_csv)

        def quantiles(df):
            if ("events" in df.columns) and ("et_min" in df.columns):
                ev = df["events"].fillna(0).astype(int) == 1
                et = pd.to_numeric(df["et_min"], errors="coerce")
                et_ev = et[ev]
                q25 = float(et_ev.quantile(0.25)) if et_ev.size else boundaries[0]
                q50 = float(et_ev.quantile(0.50)) if et_ev.size else boundaries[1]
                q75 = float(et_ev.quantile(0.75)) if et_ev.size else boundaries[2]
                return (q25, q50, q75)
            return boundaries

        q_tr = quantiles(df_tr)
        q_ex = quantiles(df_ex)
        # 权重按事件样本量设定
        n_tr = (
            int((df_tr.get("events", pd.Series()).fillna(0).astype(int) == 1).sum())
            if "events" in df_tr.columns
            else 0
        )
        n_ex = (
            int((df_ex.get("events", pd.Series()).fillna(0).astype(int) == 1).sum())
            if "events" in df_ex.columns
            else 0
        )
        w_tr = n_tr / max(n_tr + n_ex, 1)
        w_ex = 1.0 - w_tr
        b1 = w_tr * q_tr[0] + w_ex * q_ex[0]
        b2 = w_tr * q_tr[1] + w_ex * q_ex[1]
        b3 = w_tr * q_tr[2] + w_ex * q_ex[2]

        # 边界取5分钟为步长的四舍五入
        def round5(x):
            return float(int(round(x / 5.0)) * 5)

        boundaries = (round5(b1), round5(b2), round5(b3))
        # 保存到临时配置文件，便于复现
        ts = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.save_dir, exist_ok=True)
        out_cfg = os.path.join(args.save_dir, f"{ts}_boundaries.yaml")
        with open(out_cfg, "w", encoding="utf-8") as f:
            f.write("boundaries:\n")
            f.write(f"  b1: {boundaries[0]}\n")
            f.write(f"  b2: {boundaries[1]}\n")
            f.write(f"  b3: {boundaries[2]}\n")
    columns = cfg.get("columns") if cfg else None
    missing_threshold = cfg.get("missing_threshold", 0.4)
    gap_threshold = cfg.get("gap_threshold", 15)
    if args.model == "lgbm":
        X_tr, y_tr = load_and_build_features(
            train_csv,
            boundaries=boundaries,
            columns=columns,
            missing_threshold=missing_threshold,
            gap_threshold=gap_threshold,
        )
        if args.external_csv:
            X_ext, y_ext = load_and_build_features(
                args.external_csv,
                boundaries=boundaries,
                columns=columns,
                missing_threshold=missing_threshold,
                gap_threshold=gap_threshold,
            )
            seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else [])]
            out = run_train_external(
                X_tr,
                y_tr,
                X_ext,
                y_ext,
                save_dir=args.save_dir,
                model_type="lgbm",
                boundaries=boundaries,
                calibrate=(args.calibration != "off"),
                calibration_method=args.calibration,
                temp=args.temp,
                clip_min=args.clip_min,
                clip_max=args.clip_max,
                scaler=args.scaler,
                cv_folds=args.cv_folds,
                seeds=seeds,
                ext_min_thresholds=(args.ext_min_thresholds.split(",") if args.ext_min_thresholds else None),
                coral=(args.coral == "on"),
                floor_eps=(args.floor_eps.split(",") if args.floor_eps else None),
                mix_alpha=args.mix_alpha,
            )
        else:
            seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else [])]
            out = run_once(
                X_tr,
                y_tr,
                save_dir=args.save_dir,
                model_type="lgbm",
                boundaries=boundaries,
                calibrate=(args.calibration != "off"),
                calibration_method=args.calibration,
                cv_folds=args.cv_folds,
                seeds=seeds,
            )
    else:
        X_seq_tr, X_static_tr, y_tr, durations_tr, events_tr = load_seq_static_survival(
            train_csv,
            boundaries=boundaries,
            columns=columns,
            missing_threshold=missing_threshold,
            gap_threshold=gap_threshold,
        )
        if args.external_csv:
            X_seq_ext, X_static_ext, y_ext, durations_ext, events_ext = (
                load_seq_static_survival(
                    args.external_csv,
                    boundaries=boundaries,
                    columns=columns,
                    missing_threshold=missing_threshold,
                    gap_threshold=gap_threshold,
                )
            )
            out = run_train_external(
                None,
                y_tr,
                None,
                y_ext,
                save_dir=args.save_dir,
                model_type="transformer",
                X_seq_train=X_seq_tr,
                X_static_train=X_static_tr,
                X_seq_ext=X_seq_ext,
                X_static_ext=X_static_ext,
                survival=(args.survival == "on"),
                boundaries=boundaries,
                durations=durations_tr,
                events=events_tr,
                coral=(args.coral == "on"),
            )
        else:
            out = run_once(
                None,
                y_tr,
                save_dir=args.save_dir,
                model_type="transformer",
                X_seq=X_seq_tr,
                X_static=X_static_tr,
                survival=(args.survival == "on"),
                boundaries=boundaries,
                durations=durations_tr,
                events=events_tr,
            )
    print(f"输出目录: {out}")


if __name__ == "__main__":
    main()
