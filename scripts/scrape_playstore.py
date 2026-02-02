import argparse
import time
import pandas as pd
from google_play_scraper import reviews, Sort

def scrape(app_id: str, n: int = 5000, lang: str = "id", country: str = "id", sleep: float = 0.25) -> pd.DataFrame:
    """Scrape ulasan Google Play memakai google-play-scraper."""
    rows = []
    token = None
    while len(rows) < n:
        batch = min(200, n - len(rows))
        result, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=batch,
            continuation_token=token
        )
        if not result:
            break
        rows.extend(result)
        if token is None:
            break
        time.sleep(sleep)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Scrape ulasan Google Play Store ke CSV.")
    parser.add_argument("--app_id", default="com.whatsapp", help="App ID di Play Store, contoh: com.whatsapp")
    parser.add_argument("--n", type=int, default=10000, help="Jumlah ulasan yang ingin diambil")
    parser.add_argument("--out", default="dataset_raw.csv", help="Path output CSV")
    parser.add_argument("--lang", default="id", help="Bahasa, contoh: id")
    parser.add_argument("--country", default="id", help="Negara, contoh: id")
    parser.add_argument("--sleep", type=float, default=0.25, help="Delay antar request (detik)")
    args = parser.parse_args()

    df = scrape(args.app_id, n=args.n, lang=args.lang, country=args.country, sleep=args.sleep)
    df.to_csv(args.out, index=False)
    print(f"saved: {len(df)} -> {args.out}")

if __name__ == "__main__":
    main()
