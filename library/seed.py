"""
Seed the library with CUBE_CONFIG resources for the sample e-commerce dataset.

Every measure and dimension carries a `description` — these are what the agent
reads from get_cube_metadata to map natural language onto the right field, so
they include common synonyms on purpose (e.g. revenue / sales / earnings).

Run:
  python seed.py                     # create configs (fresh install)
  python seed.py --update            # upsert: patch existing configs by name
  python seed.py --url http://host   # target a different library API
"""
import argparse, json, sys
import urllib.request, urllib.error

CUBE_CONFIGS = [
    {
        "name": "orders",
        "data": {
            "sql":    "SELECT * FROM orders",
            "name":   "orders",
            "public": True,
            "description": "Customer orders — one row per order, with amount, status, country and date.",
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
            "public": True,
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
            "public": True,
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
            "sql":    "SELECT oi.*, p.name as product_name, p.category FROM order_items oi JOIN products p ON p.id = oi.product_id",
            "name":   "order_items",
            "public": True,
            "description": "Order line items — one row per product within an order, joined with product info.",
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
                "category": {
                    "sql":   "category",
                    "type":  "string",
                    "title": "Category",
                    "description": "Product category of the line item. Synonyms: product type, department."
                },
                "quantity": {
                    "sql":   "quantity",
                    "type":  "number",
                    "title": "Quantity",
                    "description": "Units purchased in the line item."
                },
                "product_name": {
                    "sql":   "product_name",
                    "type":  "string",
                    "title": "Product Name",
                    "description": "Name of the product in the line item. Synonyms: item, product title."
                }
            }
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


def _existing_by_name(base: str) -> dict:
    """Map cube name -> id for configs already in the library."""
    result = _request(f"{base}/v1/CUBE_CONFIG", None, "GET")
    return {c["name"]: c["id"] for c in result.get("data", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:3001")
    parser.add_argument("--update", action="store_true",
                        help="Upsert: patch existing configs (by name) instead of creating duplicates.")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    endpoint = f"{base}/v1/CUBE_CONFIG"
    existing = _existing_by_name(base) if args.update else {}

    verb = "Upserting" if args.update else "Seeding"
    print(f"{verb} cube configs → {endpoint}\n")
    for cfg in CUBE_CONFIGS:
        name = cfg["name"]
        if args.update and name in existing:
            cid = existing[name]
            print(f"  Updating '{name}' (id={cid}) ...", end=" ")
            result = _request(f"{endpoint}/{cid}", {"name": name, "data": cfg["data"]}, "PUT")
        else:
            print(f"  Creating '{name}' ...", end=" ")
            result = _request(endpoint, cfg, "POST")
        print(f"OK (id={result['id']})" if result.get("id") else "FAILED")

    print("\nDone. Reload the Cube schema so descriptions reach /meta.")


if __name__ == "__main__":
    main()
