# -*- coding: utf-8 -*-
"""
london_zolo_scraper.py

用途：
    從 Zolo.ca 抓取安大略省倫敦市（London, Ontario）的房產列表資料，
    涵蓋 house（獨立屋）、townhouse（連棟屋）、condo（公寓）三種房型，
    供後續 K-Means 分群分析、OLS 迴歸分析使用。

    本版本新增三個欄位，皆可直接從列表頁卡片本身取得，不需要額外對每筆
    物件的詳細頁面發送請求：
        - bedrooms：房間數（從 "3 bed"、"4+2 bed" 這類文字解析出總房間數）
        - house_age_years：屋齡估計（從 "31-50 Years Old"、"Built in 1951"、
          "New" 這類文字換算成年數，此欄位在部分卡片上並未提供，缺失時為
          None）
        - distance_to_western_km：物件與西安大略大學（Western University）
          主校區的直線距離（公里），利用卡片中已內嵌的經緯度
          （itemprop="geo"）搭配 Haversine 公式計算，同樣不需要額外請求。

    使用者原本列出的其餘因素（車位、地下室有無、裝潢狀況、所在道路最高
    時速、土地用途分區、當年利率水準、當年經濟成長率、當年人口移入速度）
    並未加入，原因如下：
        - 車位、地下室有無、裝潢狀況：Zolo 列表頁卡片沒有這些資訊，只有
          點進每筆物件的詳細頁面才看得到，若要抓取須針對每筆資料再多發
          一次請求，會讓總請求量翻倍，對伺服器負擔與抓取時間影響較大，
          裝潢狀況更牽涉主觀判斷，難以用固定規則穩定解析。
        - 所在道路最高時速、土地用途分區：這兩項資料不在 Zolo 上，需要另外
          串接市政府的地理資訊圖層（GIS），並將每筆房屋座標與圖層做空間
          對位查詢，屬於獨立的資料工程，不適合用「最小新增」的方式塞進
          這支爬蟲。
        - 當年利率水準、當年經濟成長率、當年人口移入速度：這些是總體經濟
          年度指標，而這支爬蟲每次執行抓到的都是「同一個時間點」正在銷售
          的物件，所有列同一年份對應到的總體指標會是同一個數值，在橫斷面
          資料中等於常數欄位，對迴歸模型沒有解釋力，除非未來改成長期蒐集
          不同年份的成交資料，才適合加入。

資料來源頁面結構（撰寫本腳本時人工檢查過的實際 HTML 結構）：
    - 每筆房屋資料位於 <section id="gallery"> 底下的
      <article class="card-listing"> 區塊中。
    - 街道地址在 <span class="street">，次分區（如 London South）在
      <span class="city">，省份在 <span class="province">。
    - 鄰里名稱（如 "South K"）在 <span class="neighbourhood">，文字前面
      會帶一個項目符號 "•"，需要另外去除。
    - 價格在 <li class="price"> 底下 itemprop="price" 的標籤，其 value
      屬性即為不含逗號的原始數字（例如 value="799000"）。
    - 房型名稱（House / Townhouse / Condo）在綠色「For Sale」標籤內的
      <svg><title>...</title></svg>。
    - 分頁採用網址路徑加上 "/page-2"、"/page-3" 的形式；頁面下方若還有
      下一頁，會出現 aria-label="next page of results" 的連結，沒有這個
      連結就代表已經到最後一頁。

如果之後 Zolo 改版導致上述結構跟預期不同（例如抓不到任何資料），本腳本
會直接拋出例外並中止，不會自動改抓其他網站，請人工確認頁面結構後再調整。
"""

import math
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ------------------------------------------------------------------
# 基本設定
# ------------------------------------------------------------------

BASE_URL = "https://www.zolo.ca"

# Zolo 針對不同房型有各自的清單頁面路徑，之後會逐一走訪這三種房型，
# 讓最終資料同時包含 house、townhouse、condo，符合分群分析所需的多樣性。
PROPERTY_TYPE_PATHS = {
    "house": "london-real-estate/houses",
    "townhouse": "london-real-estate/townhouses",
    "condo": "london-real-estate/condos",
}

# 模擬一般瀏覽器（Chrome）發送請求時的標頭，避免使用 requests 預設的
# User-Agent 而被網站判定為機器人擋掉。
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Western University 主校區座標，用來計算每筆房屋與學校的直線距離。
# 座標來源：公開地圖資料，供 distance_to_western_km 欄位計算使用。
WESTERN_UNIVERSITY_LAT = 43.0096
WESTERN_UNIVERSITY_LON = -81.2737

MIN_TOTAL_RECORDS = 300     # 最少要蒐集到的總筆數
MAX_TOTAL_RECORDS = 500     # 最多蒐集到這個筆數之後就停止繼續抓取
MAX_PAGES_PER_TYPE = 20     # 單一房型最多翻幾頁，避免網站異常時無限迴圈
REQUEST_TIMEOUT_SEC = 15    # 單次 HTTP 請求的逾時秒數
SLEEP_RANGE_SEC = (2, 3)    # 每次請求之間的隨機延遲秒數區間，降低對伺服器的負擔

# 輸出的 CSV 固定放在這個資料夾底下，不再用 parents[N] 反推專案根目錄。
OUTPUT_DIR = Path(r"C:\Users\flyin\OneDrive\桌面\新代碼\Linear_OLS_Regression\data")
OUTPUT_PATH = OUTPUT_DIR / "london_zolo_house_prices.csv"


def build_page_url(property_path: str, page_number: int) -> str:
    """依照房型路徑與頁碼，組出該頁的完整網址。第一頁沒有 page-N 後綴。"""
    if page_number == 1:
        return f"{BASE_URL}/{property_path}"
    return f"{BASE_URL}/{property_path}/page-{page_number}"


def fetch_html(url: str) -> str:
    """對指定網址發送 GET 請求，回傳網頁原始 HTML 字串。"""
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()  # 若回應狀態碼非 2xx，直接拋出例外讓上層處理
    return response.text


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine 公式：利用兩點的經緯度，計算球面上的直線距離（單位：公里）。
    這裡假設地球為完美球體，半徑取 6371 公里，對城市內部的距離計算而言
    誤差可忽略。
    """
    radius_km = 6371.0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (math.sin(delta_lat / 2) ** 2
         + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return radius_km * c


def parse_bed_count(text: str):
    """
    將 "3 bed"、"4+2 bed" 這類文字解析成總房間數（整數）。
    "4+2" 代表「主樓層4房 + 地下室或其他樓層2房」，這裡直接加總視為總房間數。
    解析失敗（例如完全沒有數字）則回傳 None。
    """
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return None
    return sum(int(n) for n in numbers)


def parse_house_age_years(text: str, current_year: int):
    """
    將屋齡相關文字換算成年數（整數），共處理三種實際觀察到的格式：
        - "31-50 Years Old" 這類區間格式：取區間中點（(31+50)/2 = 40.5 取40）
        - "Built in 1951" 這類建造年份格式：年數 = 目前年份 - 建造年份
        - "New"：視為屋齡0年
    如果文字不符合以上任何格式，回傳 None，代表這筆資料沒有屋齡資訊。
    """
    if text is None:
        return None

    text = text.strip()

    if text.lower() == "new":
        return 0

    range_match = re.match(r"^(\d+)-(\d+)\s*Years Old$", text)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        return (low + high) // 2

    plus_match = re.match(r"^(\d+)\+\s*Years Old$", text)
    if plus_match:
        return int(plus_match.group(1))

    built_match = re.match(r"^Built in (\d{4})$", text)
    if built_match:
        return current_year - int(built_match.group(1))

    return None


def parse_listing_cards(html: str):
    """
    解析單一列表頁面的 HTML，擷取每張房屋卡片（article.card-listing）的資料。

    回傳值：
        records: 該頁擷取到的房屋資料 list（每筆為一個 dict）
        has_next_page: 是否還有下一頁可以繼續抓取
    """
    soup = BeautifulSoup(html, "html.parser")

    gallery = soup.select_one("section#gallery")
    if gallery is None:
        # 找不到列表容器，代表頁面結構可能已經改變，直接中止讓使用者確認，
        # 不自行嘗試其他選擇器或改抓別的網站。
        raise RuntimeError(
            "找不到 <section id=\"gallery\">，Zolo 頁面結構可能已經改變，"
            "請人工確認實際頁面內容後再繼續。"
        )

    cards = gallery.select("article.card-listing")

    records = []
    for card in cards:
        street_tag = card.select_one("span.street")
        city_tag = card.select_one("span.city")
        province_tag = card.select_one("span.province")
        price_tag = card.select_one("li.price [itemprop='price']")
        type_tag = card.select_one("div.fill-green svg title")
        neighbourhood_tag = card.select_one("span.neighbourhood")
        link_tag = card.select_one("a.tile-overlay-link")
        latitude_tag = card.select_one("[itemprop='geo'] meta[itemprop='latitude']")
        longitude_tag = card.select_one("[itemprop='geo'] meta[itemprop='longitude']")
        # 房間數、屋齡都放在同一個 ul.card-listing--values 底下的 li 裡，
        # 但每張卡片實際擁有的 li 數量不固定（例如缺少屋齡資訊時只有4個），
        # 因此改用文字關鍵字比對，而不是用固定的第幾個 li 去抓。
        value_items = card.select("ul.card-listing--values li")

        # 街道地址、價格、房型是判斷「這是一筆正常房屋資料」的最低要求。
        # 實測發現 Zolo 有時會在同一頁的清單格線中，針對「已經出現過的同一筆
        # 物件」再插入一張精簡版卡片（網址路徑略有不同、且沒有綠色房型/
        # For Sale 標籤），內容其實是同一間房屋的重複資料，只是缺少
        # div.fill-green 的房型標籤。因此這裡把 type_tag 缺失也視為不完整
        # 卡片一併跳過，避免同一間房屋被重複計入。
        if street_tag is None or price_tag is None or type_tag is None:
            continue

        street = street_tag.get_text(strip=True)
        city = city_tag.get_text(strip=True) if city_tag else ""
        province = province_tag.get_text(strip=True) if province_tag else ""
        full_address = ", ".join(part for part in [street, city, province] if part)

        # neighbourhood 欄位文字前面會帶一個項目符號「•」，例如 "•South K"，
        # 這裡用 lstrip 去除符號後再取乾淨的鄰里名稱。
        neighbourhood = ""
        if neighbourhood_tag is not None:
            neighbourhood = neighbourhood_tag.get_text(strip=True).lstrip("•").strip()

        # 價格的原始數字放在 itemprop="price" 標籤的 value 屬性中
        # （例如 value="799000"，對應畫面上顯示的 "799,000"）。
        price_value = price_tag.get("value")
        price = int(price_value) if price_value and price_value.isdigit() else None

        property_type = type_tag.get_text(strip=True)
        listing_url = link_tag.get("href") if link_tag else None

        # bed_text 例如 "3 bed"、"4+2 bed"；age_text 例如 "31-50 Years Old"、
        # "Built in 1951"、"New"，缺少時維持 None，讓 parse 函式回傳 None。
        bed_text = next((li.get_text(strip=True) for li in value_items
                          if "bed" in li.get_text(strip=True).lower()), None)
        age_text = next((li.get_text(strip=True) for li in value_items
                          if "years old" in li.get_text(strip=True).lower()
                          or li.get_text(strip=True).lower().startswith("built in")
                          or li.get_text(strip=True) == "New"), None)

        bedrooms = parse_bed_count(bed_text) if bed_text else None
        house_age_years = parse_house_age_years(age_text, datetime.now().year)

        # 經緯度直接內嵌在卡片的 itemprop="geo" 區塊中，不需要額外的地理編碼
        # （geocoding）請求，就能算出這筆房屋與 Western University 的直線距離。
        distance_to_western_km = None
        if latitude_tag is not None and longitude_tag is not None:
            try:
                lat = float(latitude_tag.get("content"))
                lon = float(longitude_tag.get("content"))
                distance_to_western_km = round(
                    haversine_distance_km(lat, lon, WESTERN_UNIVERSITY_LAT, WESTERN_UNIVERSITY_LON), 2
                )
            except (TypeError, ValueError):
                # 座標格式異常時直接留空，不讓單一筆資料的座標問題中斷整個爬蟲。
                distance_to_western_km = None

        records.append(
            {
                "address": full_address,
                "district": city,  # Zolo 頁面上的 city 欄位實際上是倫敦市內的次分區（例如 London South）
                "neighbourhood": neighbourhood,
                "property_type": property_type,
                "bedrooms": bedrooms,
                "house_age_years": house_age_years,
                "distance_to_western_km": distance_to_western_km,
                "price": price,
                "listing_url": listing_url,
            }
        )

    # 頁面下方若還有下一頁，會出現 aria-label="next page of results" 的連結。
    # 注意：這個分頁連結在實際解析後的 DOM 樹中並不屬於
    # <section id="gallery"> 的子孫節點（雖然在原始 HTML 文字中看起來緊接在
    # gallery 內容之後），因此必須在整個頁面（soup）範圍內尋找，而不是只在
    # gallery 範圍內找，否則會誤判為「已無下一頁」而提早停止翻頁。
    has_next_page = soup.select_one('a[aria-label="next page of results"]') is not None

    return records, has_next_page


def scrape_property_type(label: str, property_path: str, target_count: int):
    """
    針對單一房型（house / townhouse / condo）逐頁抓取資料，
    直到累計筆數達到 target_count、沒有下一頁、或超過安全頁數上限為止。
    """
    collected = []
    page_number = 1

    while len(collected) < target_count and page_number <= MAX_PAGES_PER_TYPE:
        url = build_page_url(property_path, page_number)
        print(f"[{label}] 正在抓取第 {page_number} 頁：{url}")

        html = fetch_html(url)
        records, has_next_page = parse_listing_cards(html)

        if page_number == 1 and not records:
            # 第一頁就抓不到任何房屋卡片，很可能是頁面結構已經改變，
            # 直接中止並回報，不自行更換抓取目標。
            raise RuntimeError(
                f"[{label}] 第一頁沒有抓到任何房屋卡片，"
                "Zolo 頁面結構可能已經改變，請人工確認後再繼續。"
            )

        collected.extend(records)
        print(f"[{label}] 第 {page_number} 頁擷取 {len(records)} 筆，累計 {len(collected)} 筆")

        if not has_next_page:
            print(f"[{label}] 已經是最後一頁，停止翻頁")
            break

        page_number += 1

        # 禮貌性延遲：每次請求之間隨機停頓 2~3 秒，避免短時間內對伺服器
        # 發送過多請求。
        time.sleep(random.uniform(*SLEEP_RANGE_SEC))

    return collected


def main():
    all_records = []

    # 三種房型平均分配抓取目標，讓最終總筆數落在
    # MIN_TOTAL_RECORDS ~ MAX_TOTAL_RECORDS 之間。
    target_per_type = MAX_TOTAL_RECORDS // len(PROPERTY_TYPE_PATHS)

    for label, property_path in PROPERTY_TYPE_PATHS.items():
        records = scrape_property_type(label, property_path, target_per_type)
        all_records.extend(records)

        # 換下一種房型之前也稍作停頓，維持一致的請求節奏。
        time.sleep(random.uniform(*SLEEP_RANGE_SEC))

    # 若三種房型都抓完後，總筆數仍不足最低需求，就針對抓取結果最豐富的
    # house 類型再多抓幾頁補足（Zolo 上 house 的物件數量最多，最適合補量）。
    if len(all_records) < MIN_TOTAL_RECORDS:
        shortfall = MIN_TOTAL_RECORDS - len(all_records)
        print(f"目前僅蒐集 {len(all_records)} 筆，低於最低需求，針對 house 類型再補抓約 {shortfall} 筆")
        extra_records = scrape_property_type(
            "house_extra", PROPERTY_TYPE_PATHS["house"], target_per_type + shortfall
        )
        all_records.extend(extra_records)

    df = pd.DataFrame(all_records)

    # 依照房屋詳細頁網址去除重複資料，避免補抓或分頁過程中重複收錄同一筆房屋。
    if "listing_url" in df.columns:
        df = df.drop_duplicates(subset="listing_url").reset_index(drop=True)

    # 保險機制：Zolo 有時會用「不同的網址路徑」重複顯示同一間房屋
    # （例如同一筆物件同時出現在 /london-real-estate/... 與
    # /london-east-real-estate/... 這類不同子分區前綴的網址下），
    # 單純用 listing_url 去重無法抓到這種情況。這裡改用「門牌號碼（去除
    # 樓層/單位前綴）＋ 價格」作為第二層去重依據，避免同一間房屋在最終
    # 資料集中被重複計入，影響後續分群結果。
    address_number = df["address"].str.split(",").str[0].str.replace(
        r"^[^-]*-", "", regex=True
    ).str.strip().str.lower()
    df = df.loc[~pd.Series(list(zip(address_number, df["price"]))).duplicated().values]
    df = df.reset_index(drop=True)

    # 若總筆數超過上限，隨機抽樣保留 MAX_TOTAL_RECORDS 筆
    # （固定 random_state 以利結果可重現）。
    if len(df) > MAX_TOTAL_RECORDS:
        df = df.sample(n=MAX_TOTAL_RECORDS, random_state=42).reset_index(drop=True)

    # 確保輸出資料夾存在，再將結果寫成 CSV。
    # 使用 utf-8-sig 編碼，讓 Excel 開啟時中文字不會亂碼。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"共蒐集 {len(df)} 筆資料，已輸出至：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
