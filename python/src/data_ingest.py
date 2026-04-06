import yfinance as yf
import numpy as np
import json
from confluent_kafka import Producer
from datetime import datetime


def get_risk_free_rate() -> float:
    """
    Fetches the current 3-month US Treasury yield as a proxy for the risk-free rate.
    default Fallback of 5%.
    aligned with HestonParameters.pure: riskFreeRate : Float[1]
    """
    try:
        tbill = yf.Ticker("^IRX")  # 13-week T-bill yield index
        rate = tbill.fast_info["lastPrice"] / 100.0
        return float(rate)
    except Exception:
        return 0.05  # default in case of exception -> 5% annulized


def get_market_seed_data(asset_id: str = "SPY"):
    """
    Fetches market data,
    derives Heston parameters,
    aligned with Legend schemas: HestonParameters.pure.
    """
    data = yf.download([asset_id, "^VIX"], period="1y").copy()
    spy = data["Close"][asset_id]
    vix = data["Close"]["^VIX"] / 100.0

    spy_ret = np.log(spy / spy.shift(1)).dropna()
    vix_ret = np.log(vix / vix.shift(1)).dropna()

    # mapping to HestonParameters.pure
    params = {
        "assetId": asset_id,  #
        "timestamp": datetime.now().isoformat(),  #
        "spotPrice": float(spy.iloc[-1]),  #
        "initV0": float(vix.iloc[-1] ** 2),  #
        "riskFreeRate": get_risk_free_rate(),
        "kappa": 2.0,  # Mean reversion speed
        "theta": float((vix**2).mean()),  # Long-term variance
        "sigma": 0.3,  # Vol-of-vol
        "rho": float(spy_ret.corr(vix_ret)),  # Price-Vol correlation
    }
    # Feller condition check -- follows isFellerSatisfied() in HestonParameters.pure
    # 2*kappa*theta / sigma^2
    feller_ratio = 2.0 * params["kappa"] * params["theta"] / (params["sigma"] ** 2)
    if feller_ratio <= 1.0:
        print("[WARN] Feller condition violated: Variance process may hit zero.")
    return params


def delivery_report(err, msg):
    """
    kafka producer callback -- called once per message produced to indicate delivery result.
    """
    if err is not None:
        print(f"[ERROR] Message delivery failed: {err}")
    else:
        print(f"[OK] Message delivered to {msg.topic()} [{msg.partition()}]")


def stream_to_kafka(bootstrap_servers: str = "localhost:9092"):
    """
    Produces market seeds to Kafka topic.
    """
    conf = {"bootstrap.servers": bootstrap_servers}
    producer = Producer(conf)
    market_params = get_market_seed_data()
    # produce message
    producer.produce(
        "finance.heston.parameters",
        key=market_params["assetId"],
        value=json.dumps(market_params),
        callback=delivery_report,
    )
    producer.flush()
    print(f"[INFO] Streamed params: {json.dumps(market_params, indent=2)}")


if __name__ == "__main__":
    print("Streaming Heston parameters to Kafka...")
    stream_to_kafka()
