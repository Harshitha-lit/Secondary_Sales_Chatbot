import json
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(__file__).resolve().parents[1]
SKU_PARQUET_PATH = ROOT_DIR / "dim_sku.parquet"
SKU_OUTPUT_PATH = FRONTEND_DIR / "public" / "sku-metadata.json"
DISTRIBUTOR_PARQUET_PATH = ROOT_DIR / "dim_distributor.parquet"
DISTRIBUTOR_OUTPUT_PATH = FRONTEND_DIR / "public" / "distributor-metadata.json"


def normalized_name(column_name: str) -> str:
    return column_name.strip().lower().replace(" ", "_")


def score_column(column_name: str, positive_terms: list[str], negative_terms: list[str] | None = None) -> int:
    normalized = normalized_name(column_name)
    score = 0

    for term in positive_terms:
        if term in normalized:
            score += 2

    if negative_terms:
        for term in negative_terms:
            if term in normalized:
                score -= 3

    return score


def pick_column(columns: list[str], positive_terms: list[str], negative_terms: list[str] | None = None) -> str | None:
    ranked = sorted(
        (
            (score_column(column_name, positive_terms, negative_terms), column_name)
            for column_name in columns
        ),
        key=lambda item: (item[0], -len(item[1])),
        reverse=True,
    )

    best_score, best_column = ranked[0]
    return best_column if best_score > 0 else None


def build_display_name(product_name: str, sku_code: str, fallback_id: str) -> str:
    if product_name and sku_code:
        return f"{product_name} ({sku_code})"
    if product_name:
        return product_name
    if sku_code:
        return sku_code
    return fallback_id


def value_as_text(row: pd.Series, column_name: str | None) -> str:
    if not column_name or column_name not in row.index:
        return ""

    value = row[column_name]
    if pd.isna(value):
        return ""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value).strip()


def main() -> None:
    if not SKU_PARQUET_PATH.exists():
        raise FileNotFoundError(f"SKU dimension not found at {SKU_PARQUET_PATH}")

    dataframe = pd.read_parquet(SKU_PARQUET_PATH)
    columns = list(dataframe.columns)

    id_column = pick_column(columns, ["sku_sk", "sku_id", "product_sk", "product_id"], ["system"])
    code_column = pick_column(columns, ["sku_code", "product_code", "item_code"], ["hsn"])
    name_column = pick_column(
        columns,
        ["sku_description", "product_name", "sku_name", "description", "name"],
        ["brand", "category", "sub_brand"],
    )
    brand_column = pick_column(columns, ["brand_name", "brand"], ["sub_brand_id"])
    category_column = pick_column(columns, ["category_name", "category"], ["sub_category", "subcategory"])
    subcategory_column = pick_column(columns, ["subcategory_name", "sub_category", "subcategory"])
    abbreviation_column = pick_column(columns, ["abbreviation", "short_name", "display_name"])
    system_id_column = pick_column(columns, ["sku_system_id", "system_id"])
    weight_column = pick_column(columns, ["weight_kg", "weight"])
    volume_column = pick_column(columns, ["volume_ltr", "volume", "ltr", "litre"])
    gram_column = pick_column(columns, ["gram", "grams"])
    current_flag_column = pick_column(columns, ["is_current", "current"])
    effective_from_column = pick_column(columns, ["effective_from", "valid_from", "start_date"])

    if current_flag_column and current_flag_column in dataframe.columns:
        current_rows = dataframe[dataframe[current_flag_column] == True]
        if not current_rows.empty:
            dataframe = current_rows.copy()

    if effective_from_column and effective_from_column in dataframe.columns:
        dataframe = dataframe.sort_values(by=effective_from_column, ascending=False)

    if id_column is None:
        raise ValueError("Unable to detect a SKU identifier column from dim_sku.parquet")

    dataframe = dataframe.drop_duplicates(subset=[id_column], keep="first")

    items_by_id: dict[str, dict[str, str]] = {}
    for _, row in dataframe.iterrows():
        sku_id = value_as_text(row, id_column)
        if not sku_id:
            continue

        sku_code = value_as_text(row, code_column)
        product_name = value_as_text(row, name_column)
        abbreviation = value_as_text(row, abbreviation_column)
        brand = value_as_text(row, brand_column)
        category = value_as_text(row, category_column)
        subcategory = value_as_text(row, subcategory_column)
        system_id = value_as_text(row, system_id_column)
        weight = value_as_text(row, weight_column)
        volume = value_as_text(row, volume_column)
        gram = value_as_text(row, gram_column)

        fallback_identifier = abbreviation or sku_code or system_id or sku_id
        display_name = build_display_name(product_name, sku_code, fallback_identifier)

        items_by_id[sku_id] = {
            "skuId": sku_id,
            "skuCode": sku_code,
            "productName": product_name,
            "displayName": display_name,
            "bestIdentifier": display_name if display_name else fallback_identifier,
            "brand": brand,
            "category": category,
            "subcategory": subcategory,
            "abbreviation": abbreviation,
            "systemId": system_id,
            "weightKg": weight,
            "volumeLtr": volume,
            "gram": gram,
        }

    payload = {
        "generatedAt": pd.Timestamp.now("UTC").isoformat(),
        "detectedColumns": {
            "skuId": id_column,
            "skuCode": code_column,
            "productName": name_column,
            "brand": brand_column,
            "category": category_column,
            "subcategory": subcategory_column,
            "abbreviation": abbreviation_column,
            "systemId": system_id_column,
            "weightKg": weight_column,
            "volumeLtr": volume_column,
            "gram": gram_column,
        },
        "itemsById": items_by_id,
    }

    SKU_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SKU_OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not DISTRIBUTOR_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Distributor dimension not found at {DISTRIBUTOR_PARQUET_PATH}")

    distributor_dataframe = pd.read_parquet(DISTRIBUTOR_PARQUET_PATH)
    distributor_columns = list(distributor_dataframe.columns)

    distributor_id_column = pick_column(
        distributor_columns,
        ["distributor_sk", "distributor_id", "customer_sk", "customer_id"],
        ["zone", "subzone"],
    )
    distributor_code_column = pick_column(
        distributor_columns,
        ["warehouse_erp_code", "distributor_code", "erp_code", "customer_code", "code"],
        ["gstin", "zone", "subzone"],
    )
    distributor_name_column = pick_column(
        distributor_columns,
        ["warehouse_name", "distributor_name", "customer_name", "name"],
        ["zone", "subzone", "state", "district", "city", "wd_name"],
    )
    zone_column = pick_column(distributor_columns, ["zone_name", "zone"], ["zone_id", "subzone"])
    subzone_column = pick_column(
        distributor_columns,
        ["subzone_name", "sub_zone_name", "subzone"],
        ["subzone_id"],
    )
    state_column = pick_column(distributor_columns, ["state_name", "state"])
    city_column = pick_column(distributor_columns, ["warehouse_city", "city"])
    district_column = pick_column(distributor_columns, ["warehouse_district", "district"])

    if distributor_id_column is None:
        raise ValueError("Unable to detect a distributor identifier column from dim_distributor.parquet")

    distributor_dataframe = distributor_dataframe.drop_duplicates(
        subset=[distributor_id_column],
        keep="first",
    )

    distributors_by_id: dict[str, dict[str, str]] = {}
    for _, row in distributor_dataframe.iterrows():
        distributor_id = value_as_text(row, distributor_id_column)
        if not distributor_id:
            continue

        distributor_code = value_as_text(row, distributor_code_column)
        distributor_name = value_as_text(row, distributor_name_column)
        zone_name = value_as_text(row, zone_column)
        subzone_name = value_as_text(row, subzone_column)
        state_name = value_as_text(row, state_column)
        city_name = value_as_text(row, city_column)
        district_name = value_as_text(row, district_column)

        if distributor_name and distributor_code:
            display_name = f"{distributor_name} ({distributor_code})"
        elif distributor_name:
            display_name = distributor_name
        elif distributor_code:
            display_name = distributor_code
        else:
            display_name = distributor_id

        distributors_by_id[distributor_id] = {
            "distributorId": distributor_id,
            "distributorCode": distributor_code,
            "distributorName": distributor_name,
            "displayName": display_name,
            "bestIdentifier": display_name,
            "zoneName": zone_name,
            "subzoneName": subzone_name,
            "stateName": state_name,
            "cityName": city_name,
            "districtName": district_name,
        }

    distributor_payload = {
        "generatedAt": pd.Timestamp.now("UTC").isoformat(),
        "detectedColumns": {
            "distributorId": distributor_id_column,
            "distributorCode": distributor_code_column,
            "distributorName": distributor_name_column,
            "zoneName": zone_column,
            "subzoneName": subzone_column,
            "stateName": state_column,
            "cityName": city_column,
            "districtName": district_column,
        },
        "itemsById": distributors_by_id,
    }

    DISTRIBUTOR_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DISTRIBUTOR_OUTPUT_PATH.write_text(
        json.dumps(distributor_payload, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
