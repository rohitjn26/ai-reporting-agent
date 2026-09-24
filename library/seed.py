"""
Seed the library with CUBE_CONFIG resources for the sample e-commerce dataset.

Every measure and dimension carries a `description` — these are what the agent
reads from get_cube_metadata to map natural language onto the right field, so
they include common synonyms on purpose (e.g. revenue / sales / earnings).

Run:
  python seed.py                     # create configs (fresh install)
  python seed.py --update            # upsert: patch existing configs by name
  python seed.py --url http://host   # target a different library API
  python seed.py --from-draft semantic_layer.json   # load a Schema-tab export:
                                     # CSVs -> Postgres schema <dataset>, then cubes + views
"""
import argparse, json, sys
from pathlib import Path
import urllib.request, urllib.error

CUBE_CONFIGS = [
    {
        "name": "orders",
        "data": {
            "sql":    "SELECT * FROM orders",
            "name":   "orders",
            "public": False,
            "description": "Customer orders — one row per order, with amount, status, country and date.",
            "joins": {
                "customers": {
                    "sql":          "${CUBE}.customer_id = ${customers.id}",
                    "relationship": "many_to_one"
                }
            },
            "measures": {
                "count": {
                    "sql":   "id",
                    "type":  "count",
                    "title": "Order Count",
                    "description": "Number of orders placed. Synonyms: order count, how many orders, number of purchases."
                },
                "day_count": {
                    "sql":   "DATE(created_at)",
                    "type":  "count_distinct",
                    "title": "Day Count",
                    "description": "Number of distinct calendar days on which orders were placed (active order days)."
                },
                "total_revenue": {
                    "sql":   "total_amount",
                    "type":  "sum",
                    "title": "Total Revenue",
                    "description": "Total order revenue in USD (sum of order amounts). Synonyms: revenue, sales, income, earnings, turnover, money made."
                },
                "avg_order_value": {
                    "sql":   "total_amount",
                    "type":  "avg",
                    "title": "Avg Order Value",
                    "description": "Average order value (AOV) in USD — mean amount per order. Synonyms: average spend, typical order size."
                }
            },
            "dimensions": {
                "id": {
                    "sql":   "id",
                    "type":  "number",
                    "title": "Order ID",
                    "primary_key": True,
                    "description": "Unique order identifier (primary key)."
                },
                "status": {
                    "sql":   "status",
                    "type":  "string",
                    "title": "Status",
                    "description": "Order lifecycle status: pending, completed, cancelled, or refunded."
                },
                "country": {
                    "sql":   "country",
                    "type":  "string",
                    "title": "Country",
                    "description": "Country where the order was placed. Synonyms: region, market, geography, location."
                },
                "bi_monthly": {
                    "sql":   "CASE WHEN EXTRACT(MONTH FROM created_at) IN (1,2) THEN 'Jan-Feb' WHEN EXTRACT(MONTH FROM created_at) IN (3,4) THEN 'Mar-Apr' ELSE 'Other' END",
                    "type":  "string",
                    "title": "Bi-Monthly",
                    "description": "Two-month bucket of the order date (Jan-Feb, Mar-Apr, else Other)."
                },
                "created_at": {
                    "sql":   "created_at",
                    "type":  "time",
                    "title": "Order Date",
                    "description": "Timestamp the order was placed. Use for time trends by day/week/month/quarter/year. Synonyms: order date, when ordered, over time."
                },
                "day_of_week": {
                    "sql":   "TO_CHAR(created_at, 'Day')",
                    "type":  "string",
                    "title": "Day of Week",
                    "description": "Weekday name the order was placed (Monday–Sunday). Synonyms: weekday."
                },
                "order_bucket": {
                    "sql":   "CONCAT(((FLOOR((day_count - 1) / 5) * 5) + 1)::int, '-', ((FLOOR((day_count - 1) / 5) * 5) + 5)::int)",
                    "type":  "string",
                    "title": "Order Bucket",
                    "description": "Active-order-days grouped into bands of 5 (e.g. '1-5', '6-10')."
                },
                "revenue_bucket": {
                    "sql":   "CONCAT('$', (FLOOR(total_amount / 100) * 100)::int, ' - $', (FLOOR(total_amount / 100) * 100 + 99)::int)",
                    "type":  "string",
                    "title": "Revenue Bucket",
                    "description": "Order amount bucketed into $100 bands (e.g. '$0 - $99', '$100 - $199')."
                }
            }
        }
    },
    {
        "name": "products",
        "data": {
            "sql":    "SELECT * FROM products",
            "name":   "products",
            "public": False,
            "description": "Product catalog — one row per product, with category and price.",
            "measures": {
                "count": {
                    "sql":   "id",
                    "type":  "count",
                    "title": "Product Count",
                    "description": "Number of products in the catalog. Synonyms: how many products, number of items."
                },
                "avg_price": {
                    "sql":   "price",
                    "type":  "avg",
                    "title": "Avg Price",
                    "description": "Average product price in USD. Synonyms: mean price, typical price."
                },
                "max_price": {
                    "sql":   "price",
                    "type":  "max",
                    "title": "Max Price",
                    "description": "Highest product price in USD. Synonyms: most expensive, priciest."
                }
            },
            "dimensions": {
                "id": {
                    "sql":   "id",
                    "type":  "number",
                    "title": "Product ID",
                    "primary_key": True,
                    "description": "Unique product identifier (primary key)."
                },
                "name": {
                    "sql":   "name",
                    "type":  "string",
                    "title": "Product Name",
                    "description": "Product name. Synonyms: item, product title."
                },
                "price": {
                    "sql":   "price",
                    "type":  "number",
                    "title": "Price",
                    "description": "Product unit price in USD."
                },
                "category": {
                    "sql":   "category",
                    "type":  "string",
                    "title": "Category",
                    "description": "Product category. Synonyms: product type, department, segment."
                }
            }
        }
    },
    {
        "name": "customers",
        "data": {
            "sql":    "SELECT * FROM customers",
            "name":   "customers",
            "public": False,
            "description": "Customer base — one row per customer, with country and signup date.",
            "measures": {
                "count": {
                    "sql":   "id",
                    "type":  "count",
                    "title": "Customer Count",
                    "description": "Number of customers. Synonyms: customer count, how many customers, user count, number of buyers."
                }
            },
            "dimensions": {
                "id": {
                    "sql":   "id",
                    "type":  "number",
                    "title": "Customer ID",
                    "primary_key": True,
                    "description": "Unique customer identifier (primary key)."
                },
                "name": {
                    "sql":   "name",
                    "type":  "string",
                    "title": "Customer Name",
                    "description": "Customer full name."
                },
                "country": {
                    "sql":   "country",
                    "type":  "string",
                    "title": "Country",
                    "description": "Customer's country. Synonyms: region, market, geography, location."
                },
                "created_at": {
                    "sql":   "created_at",
                    "type":  "time",
                    "title": "Signup Date",
                    "description": "Timestamp the customer signed up. Use for signup/registration trends. Synonyms: signup date, registration date, join date."
                }
            }
        }
    },
    {
        "name": "order_items",
        "data": {
            "sql":    "SELECT * FROM order_items",
            "name":   "order_items",
            "public": False,
            "description": "Order line items — one row per product within an order.",
            "joins": {
                "orders": {
                    "sql":          "${CUBE}.order_id = ${orders.id}",
                    "relationship": "many_to_one"
                },
                "products": {
                    "sql":          "${CUBE}.product_id = ${products.id}",
                    "relationship": "many_to_one"
                }
            },
            "measures": {
                "count": {
                    "sql":   "id",
                    "type":  "count",
                    "title": "Line Item Count",
                    "description": "Number of order line items. Synonyms: number of line items."
                },
                "order_count": {
                    "sql":   "order_id",
                    "type":  "count_distinct",
                    "title": "Number of Orders",
                    "description": "Number of distinct orders represented in the line items."
                },
                "total_revenue": {
                    "sql":   "quantity * unit_price",
                    "type":  "sum",
                    "title": "Total Revenue",
                    "description": "Total line-item revenue in USD (quantity × unit price). Synonyms: revenue, sales, earnings."
                },
                "total_quantity": {
                    "sql":   "quantity",
                    "type":  "sum",
                    "title": "Total Quantity Sold",
                    "description": "Total units sold across line items. Synonyms: units sold, quantity sold, volume."
                },
                "avg_order_value": {
                    "sql":   "SUM(quantity * unit_price) / COUNT(DISTINCT order_id)",
                    "type":  "number",
                    "title": "Average Order Value",
                    "description": "Average revenue per order derived from line items (AOV in USD)."
                },
                "revenue_per_item": {
                    "sql":   "ROUND(SUM(quantity * unit_price) / SUM(quantity), 2)",
                    "type":  "number",
                    "title": "Revenue Per Item",
                    "description": "Average revenue per unit sold in USD (revenue ÷ units). Synonyms: revenue per unit, price realised."
                }
            },
            "dimensions": {
                "id": {
                    "sql":   "id",
                    "type":  "number",
                    "title": "Item ID",
                    "primary_key": True,
                    "description": "Unique line-item identifier (primary key)."
                },
                "quantity": {
                    "sql":   "quantity",
                    "type":  "number",
                    "title": "Quantity",
                    "description": "Units purchased in the line item."
                }
                # product_name & category now come from the products cube via the
                # product_sales view (no longer denormalised into this cube's SQL).
            }
        }
    }
]


# Views are the ONLY public surface the agent sees (base cubes above are public:False).
# A query can never span two views, so each view is a self-contained analytical area.
# Member descriptions (with synonyms) live on the `includes` objects so they reach /meta.
VIEW_CONFIGS = [
    {
        "name": "sales",
        "data": {
            "name": "sales",
            "public": True,
            "description": "Order & customer analytics — revenue, order counts and status by country, date and customer. One row per order.",
            "cubes": [
                {
                    "join_path": "orders",
                    "includes": [
                        "count", "day_count", "total_revenue", "avg_order_value",
                        "status", "country", "bi_monthly", "created_at",
                        "day_of_week", "order_bucket", "revenue_bucket"
                    ]
                },
                {
                    "join_path": "orders.customers",
                    "includes": [
                        {"name": "name", "alias": "customer_name",
                         "description": "Name of the customer who placed the order."},
                        {"name": "country", "alias": "customer_country",
                         "description": "Home country of the customer (distinct from the order's country). Synonyms: buyer country, customer region."}
                    ]
                }
            ]
        }
    },
    {
        "name": "product_sales",
        "data": {
            "name": "product_sales",
            "public": True,
            "description": "Line-item & product analytics — units sold, line revenue and revenue per item by product, category and order date. One row per order line item.",
            "cubes": [
                {
                    "join_path": "order_items",
                    "includes": [
                        "count", "order_count", "total_revenue", "total_quantity",
                        "avg_order_value", "revenue_per_item", "quantity"
                    ]
                },
                {
                    "join_path": "order_items.products",
                    "includes": [
                        {"name": "name", "alias": "product_name",
                         "description": "Name of the product sold. Synonyms: item, product title."},
                        {"name": "category",
                         "description": "Product category. Synonyms: product type, department, segment."},
                        {"name": "price", "alias": "unit_list_price",
                         "description": "Catalog unit price of the product in USD."}
                    ]
                },
                {
                    "join_path": "order_items.orders",
                    "includes": [
                        {"name": "created_at", "alias": "order_date",
                         "description": "Date the order was placed. Use for product-sales trends over time by day/week/month/quarter/year."},
                        {"name": "status", "alias": "order_status",
                         "description": "Order status: pending, completed, cancelled, refunded."},
                        {"name": "country", "alias": "order_country",
                         "description": "Country where the order was placed. Synonyms: market, region."}
                    ]
                }
            ]
        }
    }
]


def _request(url: str, payload: dict | None, method: str) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        print(f"  ERROR {e.code}: {e.read().decode()}")
        return {}


def _existing_by_name(base: str, resource_type: str) -> dict:
    """Map resource name -> id for resources of a type already in the library."""
    result = _request(f"{base}/v1/{resource_type}", None, "GET")
    return {c["name"]: c["id"] for c in result.get("data", [])}


def _seed(base: str, resource_type: str, configs: list, update: bool) -> None:
    endpoint = f"{base}/v1/{resource_type}"
    existing = _existing_by_name(base, resource_type) if update else {}

    verb = "Upserting" if update else "Seeding"
    print(f"{verb} {resource_type} → {endpoint}")
    for cfg in configs:
        name = cfg["name"]
        if update and name in existing:
            cid = existing[name]
            print(f"  Updating '{name}' (id={cid}) ...", end=" ")
            result = _request(f"{endpoint}/{cid}", {"name": name, "data": cfg["data"]}, "PUT")
        else:
            print(f"  Creating '{name}' ...", end=" ")
            result = _request(endpoint, {"name": name, "data": cfg["data"]}, "POST")
        print(f"OK (id={result['id']})" if result.get("id") else "FAILED")
    print()


def _draft_collisions(draft: dict, existing: dict[str, dict]) -> list[str]:
    """Draft cube/view names that already exist in the library."""
    return [f"{rtype} {cfg['name']}"
            for rtype, key in (("CUBE_CONFIG", "cubes"), ("VIEW", "views"))
            for cfg in draft.get(key, []) if cfg["name"] in existing.get(rtype, {})]


def _from_draft(base: str, path: str, force: bool, skip_data: bool, pg_url: str | None) -> None:
    """Seed a Schema-tab export: load its CSVs into Postgres, then upsert cubes + views."""
    with open(path) as f:
        draft = json.load(f)
    existing = {t: _existing_by_name(base, t) for t in ("CUBE_CONFIG", "VIEW")}
    clashes = _draft_collisions(draft, existing)
    if clashes and not force:
        sys.exit("Refusing to overwrite existing library entries (use --force):\n  "
                 + "\n  ".join(clashes))

    if not skip_data:
        # discovery/ lives at the repo root; appended so it can't shadow installed packages
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from discovery.publish import DEFAULT_PG_URL, load_data
        print(f"Loading data into Postgres schema '{draft.get('dataset')}' ...")
        for table, rows in load_data(draft, pg_url or DEFAULT_PG_URL).items():
            print(f"  {table}: {rows} rows")
        print()

    _seed(base, "CUBE_CONFIG", draft.get("cubes", []), update=True)
    _seed(base, "VIEW", draft.get("views", []), update=True)
    for note in draft.get("notes", []):
        print(f"  note: {note}")
    print("Done. Reload the Cube schema so the new views reach /meta.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:3001")
    parser.add_argument("--update", action="store_true",
                        help="Upsert: patch existing configs (by name) instead of creating duplicates.")
    parser.add_argument("--from-draft", metavar="JSON",
                        help="Seed a semantic_layer.json exported from the Schema tab instead of the built-in configs.")
    parser.add_argument("--force", action="store_true",
                        help="--from-draft: overwrite library entries that already exist by name.")
    parser.add_argument("--skip-data", action="store_true",
                        help="--from-draft: don't (re)load the CSVs into Postgres.")
    parser.add_argument("--pg-url", help="--from-draft: analytics Postgres URL "
                        "(default $ANALYTICS_DB_URL or localhost:5432/reporting).")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    if args.from_draft:
        _from_draft(base, args.from_draft, args.force, args.skip_data, args.pg_url)
        return
    # Cubes first (views reference them), then views.
    _seed(base, "CUBE_CONFIG", CUBE_CONFIGS, args.update)
    _seed(base, "VIEW", VIEW_CONFIGS, args.update)

    print("Done. Reload the Cube schema so joins/views and descriptions reach /meta.")


if __name__ == "__main__":
    main()
