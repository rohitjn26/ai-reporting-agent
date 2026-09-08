"""
Seed the library with CUBE_CONFIG resources for the sample e-commerce dataset.
Run:  python seed.py [--url http://localhost:3001]
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
            "description": "Customer orders with revenue and status",
            "measures": {
                "count": {
                    "sql":   "id",
                    "type":  "count",
                    "title": "Order Count",
                    "description": "Total number of orders"
                },
                "total_revenue": {
                    "sql":   "total_amount",
                    "type":  "sum",
                    "title": "Total Revenue",
                    "description": "Sum of all order amounts"
                },
                "avg_order_value": {
                    "sql":   "total_amount",
                    "type":  "avg",
                    "title": "Avg Order Value",
                    "description": "Average order amount"
                }
            },
            "dimensions": {
                "id": {
                    "sql":   "id",
                    "type":  "number",
                    "title": "Order ID",
                    "primary_key": True
                },
                "status": {
                    "sql":   "status",
                    "type":  "string",
                    "title": "Status",
                    "description": "Order status: pending, completed, cancelled, refunded"
                },
                "country": {
                    "sql":   "country",
                    "type":  "string",
                    "title": "Country"
                },
                "created_at": {
                    "sql":   "created_at",
                    "type":  "time",
                    "title": "Order Date"
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
            "description": "Product catalog with categories and pricing",
            "measures": {
                "count": {
                    "sql":   "id",
                    "type":  "count",
                    "title": "Product Count"
                },
                "avg_price": {
                    "sql":   "price",
                    "type":  "avg",
                    "title": "Avg Price"
                },
                "max_price": {
                    "sql":   "price",
                    "type":  "max",
                    "title": "Max Price"
                }
            },
            "dimensions": {
                "id": {
                    "sql":   "id",
                    "type":  "number",
                    "title": "Product ID",
                    "primary_key": True
                },
                "name": {
                    "sql":   "name",
                    "type":  "string",
                    "title": "Product Name"
                },
                "category": {
                    "sql":   "category",
                    "type":  "string",
                    "title": "Category"
                },
                "price": {
                    "sql":   "price",
                    "type":  "number",
                    "title": "Price"
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
            "description": "Customer base with geographic distribution",
            "measures": {
                "count": {
                    "sql":   "id",
                    "type":  "count",
                    "title": "Customer Count"
                }
            },
            "dimensions": {
                "id": {
                    "sql":   "id",
                    "type":  "number",
                    "title": "Customer ID",
                    "primary_key": True
                },
                "name": {
                    "sql":   "name",
                    "type":  "string",
                    "title": "Customer Name"
                },
                "country": {
                    "sql":   "country",
                    "type":  "string",
                    "title": "Country"
                },
                "created_at": {
                    "sql":   "created_at",
                    "type":  "time",
                    "title": "Signup Date"
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
            "description": "Line items within orders, joined with product info",
            "measures": {
                "count": {
                    "sql":   "id",
                    "type":  "count",
                    "title": "Line Item Count"
                },
                "total_quantity": {
                    "sql":   "quantity",
                    "type":  "sum",
                    "title": "Total Quantity Sold"
                },
                "total_revenue": {
                    "sql":   "quantity * unit_price",
                    "type":  "sum",
                    "title": "Total Revenue"
                }
            },
            "dimensions": {
                "id": {
                    "sql":   "id",
                    "type":  "number",
                    "title": "Item ID",
                    "primary_key": True
                },
                "product_name": {
                    "sql":   "product_name",
                    "type":  "string",
                    "title": "Product Name"
                },
                "category": {
                    "sql":   "category",
                    "type":  "string",
                    "title": "Category"
                },
                "quantity": {
                    "sql":   "quantity",
                    "type":  "number",
                    "title": "Quantity"
                }
            }
        }
    }
]


def post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ERROR {e.code}: {body}")
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:3001")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    endpoint = f"{base}/v1/CUBE_CONFIG"

    print(f"Seeding cube configs → {endpoint}\n")
    for cfg in CUBE_CONFIGS:
        print(f"  Creating '{cfg['name']}' ...", end=" ")
        result = post(endpoint, cfg)
        if result.get("id"):
            print(f"OK (id={result['id']})")
        else:
            print("FAILED")

    print("\nDone. Run `make up` to start the stack and try the agent.")


if __name__ == "__main__":
    main()
