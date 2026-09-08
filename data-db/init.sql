\c reporting;

CREATE TABLE IF NOT EXISTS customers (
  id        SERIAL PRIMARY KEY,
  name      VARCHAR(100) NOT NULL,
  email     VARCHAR(150) UNIQUE NOT NULL,
  country   VARCHAR(60) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS products (
  id       SERIAL PRIMARY KEY,
  name     VARCHAR(150) NOT NULL,
  category VARCHAR(60) NOT NULL,
  price    NUMERIC(10,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
  id           SERIAL PRIMARY KEY,
  customer_id  INTEGER NOT NULL REFERENCES customers(id),
  status       VARCHAR(20) NOT NULL CHECK (status IN ('pending','completed','cancelled','refunded')),
  total_amount NUMERIC(10,2) NOT NULL,
  country      VARCHAR(60) NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS order_items (
  id         SERIAL PRIMARY KEY,
  order_id   INTEGER NOT NULL REFERENCES orders(id),
  product_id INTEGER NOT NULL REFERENCES products(id),
  quantity   INTEGER NOT NULL,
  unit_price NUMERIC(10,2) NOT NULL
);

-- Customers
INSERT INTO customers (name, email, country, created_at) VALUES
('Alice Martin',   'alice@example.com',   'USA',     '2024-01-05 09:00:00+00'),
('Bob Chen',       'bob@example.com',     'Canada',  '2024-01-12 11:00:00+00'),
('Carol Silva',    'carol@example.com',   'Brazil',  '2024-01-20 14:00:00+00'),
('David Lee',      'david@example.com',   'UK',      '2024-02-01 08:00:00+00'),
('Eva Müller',     'eva@example.com',     'Germany', '2024-02-10 10:00:00+00'),
('Frank Dubois',   'frank@example.com',   'France',  '2024-02-18 12:00:00+00'),
('Grace Kim',      'grace@example.com',   'Korea',   '2024-03-01 09:30:00+00'),
('Hiro Tanaka',    'hiro@example.com',    'Japan',   '2024-03-10 13:00:00+00'),
('Irene Patel',    'irene@example.com',   'India',   '2024-03-22 07:00:00+00'),
('Jack Oliveira',  'jack@example.com',    'Brazil',  '2024-04-05 16:00:00+00'),
('Kate Wilson',    'kate@example.com',    'USA',     '2024-04-15 09:00:00+00'),
('Liam Brown',     'liam@example.com',    'UK',      '2024-05-01 11:00:00+00'),
('Mia Schneider',  'mia@example.com',     'Germany', '2024-05-12 10:30:00+00'),
('Noah Taylor',    'noah@example.com',    'Canada',  '2024-06-01 08:00:00+00'),
('Olivia García',  'olivia@example.com',  'Mexico',  '2024-06-15 14:00:00+00'),
('Pedro Costa',    'pedro@example.com',   'Brazil',  '2024-07-01 12:00:00+00'),
('Quinn Adams',    'quinn@example.com',   'USA',     '2024-07-20 09:00:00+00'),
('Rosa Fernández', 'rosa@example.com',    'Mexico',  '2024-08-05 10:00:00+00'),
('Sam Nguyen',     'sam@example.com',     'Vietnam', '2024-08-15 08:00:00+00'),
('Tina Rossi',     'tina@example.com',    'Italy',   '2024-09-01 11:00:00+00');

-- Products
INSERT INTO products (name, category, price) VALUES
('Laptop Pro 15',      'Electronics',  1299.00),
('Wireless Headphones','Electronics',   199.00),
('USB-C Hub',          'Electronics',    49.99),
('Mechanical Keyboard','Electronics',   129.00),
('Webcam HD',          'Electronics',    89.00),
('Running Shoes',      'Sports',        149.00),
('Yoga Mat',           'Sports',         39.99),
('Water Bottle',       'Sports',         24.99),
('Resistance Bands',   'Sports',         19.99),
('Protein Powder',     'Sports',         59.99),
('Python Cookbook',    'Books',          45.00),
('Data Science Guide', 'Books',          55.00),
('Clean Code',         'Books',          40.00),
('Design Patterns',    'Books',          50.00),
('T-Shirt Premium',    'Clothing',       29.99),
('Hoodie Classic',     'Clothing',       59.99),
('Jeans Slim',         'Clothing',       79.99),
('Jacket Outdoor',     'Clothing',      129.00),
('Coffee Beans 1kg',   'Food',           24.99),
('Organic Tea Set',    'Food',           19.99);

-- Orders (60 orders spanning Jan–Sep 2024)
INSERT INTO orders (customer_id, status, total_amount, country, created_at) VALUES
(1,  'completed',  1498.00, 'USA',     '2024-01-08 10:00:00+00'),
(2,  'completed',   248.99, 'Canada',  '2024-01-15 14:00:00+00'),
(3,  'pending',     189.00, 'Brazil',  '2024-01-22 09:00:00+00'),
(4,  'completed',  1388.00, 'UK',      '2024-02-03 11:00:00+00'),
(5,  'cancelled',    89.00, 'Germany', '2024-02-12 13:00:00+00'),
(6,  'completed',   299.97, 'France',  '2024-02-20 10:00:00+00'),
(7,  'completed',   149.00, 'Korea',   '2024-03-05 09:00:00+00'),
(8,  'refunded',    199.00, 'Japan',   '2024-03-15 14:00:00+00'),
(9,  'completed',   374.97, 'India',   '2024-03-25 08:00:00+00'),
(10, 'completed',  1348.00, 'Brazil',  '2024-04-08 12:00:00+00'),
(11, 'pending',     109.98, 'USA',     '2024-04-18 10:00:00+00'),
(12, 'completed',   279.98, 'UK',      '2024-05-03 11:00:00+00'),
(13, 'completed',   228.98, 'Germany', '2024-05-15 09:00:00+00'),
(14, 'cancelled',   129.00, 'Canada',  '2024-06-03 14:00:00+00'),
(15, 'completed',   209.98, 'Mexico',  '2024-06-18 10:00:00+00'),
(16, 'completed',   188.98, 'Brazil',  '2024-07-04 08:00:00+00'),
(17, 'completed',  1428.00, 'USA',     '2024-07-22 12:00:00+00'),
(18, 'pending',      44.98, 'Mexico',  '2024-08-07 09:00:00+00'),
(19, 'completed',   269.97, 'Vietnam', '2024-08-18 11:00:00+00'),
(20, 'completed',   294.97, 'Italy',   '2024-09-03 10:00:00+00'),
(1,  'completed',   248.99, 'USA',     '2024-01-25 14:00:00+00'),
(2,  'completed',   178.99, 'Canada',  '2024-02-08 10:00:00+00'),
(3,  'completed',  1349.00, 'Brazil',  '2024-02-22 09:00:00+00'),
(4,  'refunded',    199.00, 'UK',      '2024-03-08 11:00:00+00'),
(5,  'completed',   358.99, 'Germany', '2024-03-18 13:00:00+00'),
(6,  'completed',   129.00, 'France',  '2024-04-01 10:00:00+00'),
(7,  'completed',   309.98, 'Korea',   '2024-04-20 09:00:00+00'),
(8,  'cancelled',    59.99, 'Japan',   '2024-05-05 14:00:00+00'),
(9,  'completed',   223.99, 'India',   '2024-05-18 08:00:00+00'),
(10, 'completed',   258.98, 'Brazil',  '2024-06-05 12:00:00+00'),
(11, 'completed',  1488.99, 'USA',     '2024-06-20 10:00:00+00'),
(12, 'pending',     149.00, 'UK',      '2024-07-08 11:00:00+00'),
(13, 'completed',   348.97, 'Germany', '2024-07-25 09:00:00+00'),
(14, 'completed',   189.98, 'Canada',  '2024-08-10 14:00:00+00'),
(15, 'cancelled',    79.99, 'Mexico',  '2024-08-22 10:00:00+00'),
(16, 'completed',   328.98, 'Brazil',  '2024-09-05 08:00:00+00'),
(17, 'refunded',    249.00, 'USA',     '2024-01-30 12:00:00+00'),
(18, 'completed',   169.98, 'Mexico',  '2024-02-14 09:00:00+00'),
(19, 'completed',   429.97, 'Vietnam', '2024-03-20 11:00:00+00'),
(20, 'completed',   194.98, 'Italy',   '2024-04-25 10:00:00+00'),
(1,  'completed',  1578.00, 'USA',     '2024-05-10 14:00:00+00'),
(2,  'pending',     109.98, 'Canada',  '2024-05-28 10:00:00+00'),
(3,  'completed',   268.98, 'Brazil',  '2024-06-10 09:00:00+00'),
(4,  'completed',   378.97, 'UK',      '2024-06-25 11:00:00+00'),
(5,  'completed',   228.99, 'Germany', '2024-07-12 13:00:00+00'),
(6,  'cancelled',   199.00, 'France',  '2024-07-28 10:00:00+00'),
(7,  'completed',  1049.00, 'Korea',   '2024-08-05 09:00:00+00'),
(8,  'completed',   299.97, 'Japan',   '2024-08-20 14:00:00+00'),
(9,  'completed',   438.99, 'India',   '2024-09-01 08:00:00+00'),
(10, 'completed',   158.98, 'Brazil',  '2024-09-06 12:00:00+00'),
(11, 'refunded',    129.00, 'USA',     '2024-02-25 10:00:00+00'),
(12, 'completed',   459.97, 'UK',      '2024-03-30 11:00:00+00'),
(13, 'completed',  1329.00, 'Germany', '2024-04-30 09:00:00+00'),
(14, 'completed',   249.98, 'Canada',  '2024-05-22 14:00:00+00'),
(15, 'completed',   599.98, 'Mexico',  '2024-06-30 10:00:00+00'),
(16, 'pending',      84.98, 'Brazil',  '2024-07-15 08:00:00+00'),
(17, 'completed',   354.97, 'USA',     '2024-08-12 12:00:00+00'),
(18, 'completed',   149.99, 'Mexico',  '2024-08-28 09:00:00+00'),
(19, 'cancelled',    49.99, 'Vietnam', '2024-09-04 11:00:00+00'),
(20, 'completed',   379.97, 'Italy',   '2024-09-07 10:00:00+00');

-- Order items (representative subset)
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 1299.00), (1, 2, 1, 199.00),
(2, 2, 1, 199.00), (2, 3, 1, 49.99),
(3, 6, 1, 149.00), (3, 8, 2, 24.99), (3, 9, 2, 19.99),
(4, 1, 1, 1299.00), (4, 3, 1, 49.99), (4, 5, 1, 89.00),
(5, 5, 1, 89.00),
(6, 15, 3, 29.99), (6, 16, 1, 59.99), (6, 20, 1, 19.99),
(7, 6, 1, 149.00),
(8, 2, 1, 199.00),
(9, 7, 1, 39.99), (9, 6, 1, 149.00), (9, 10, 3, 59.99),
(10, 1, 1, 1299.00), (10, 3, 1, 49.99),
(11, 19, 2, 24.99), (11, 20, 3, 19.99),
(12, 16, 1, 59.99), (12, 15, 4, 29.99),
(13, 4, 1, 129.00), (13, 15, 2, 29.99), (13, 20, 2, 19.99),
(14, 4, 1, 129.00),
(15, 17, 1, 79.99), (15, 19, 3, 24.99), (15, 20, 2, 19.99),
(16, 11, 1, 45.00), (16, 12, 1, 55.00), (16, 13, 1, 40.00), (16, 20, 2, 19.99),
(17, 1, 1, 1299.00), (17, 4, 1, 129.00),
(18, 19, 1, 24.99), (18, 20, 1, 19.99),
(19, 11, 2, 45.00), (19, 12, 1, 55.00), (19, 13, 1, 40.00),
(20, 16, 1, 59.99), (20, 19, 4, 24.99), (20, 11, 1, 45.00),
(21, 2, 1, 199.00), (21, 3, 1, 49.99),
(22, 17, 1, 79.99), (22, 15, 1, 29.99), (22, 20, 2, 19.99),
(23, 1, 1, 1299.00), (23, 3, 1, 49.99),
(24, 2, 1, 199.00),
(25, 6, 1, 149.00), (25, 10, 2, 59.99), (25, 9, 5, 19.99),
(26, 4, 1, 129.00),
(27, 7, 1, 39.99), (27, 8, 3, 24.99), (27, 9, 5, 19.99), (27, 6, 1, 149.00),
(28, 19, 1, 59.99),
(29, 15, 3, 29.99), (29, 11, 1, 45.00), (29, 13, 1, 40.00),
(30, 16, 1, 59.99), (30, 6, 1, 149.00), (30, 15, 1, 29.99),
(31, 1, 1, 1299.00), (31, 3, 1, 49.99), (31, 5, 1, 89.00), (31, 20, 2, 19.99),
(32, 6, 1, 149.00),
(33, 11, 2, 45.00), (33, 12, 2, 55.00), (33, 13, 1, 40.00), (33, 14, 1, 50.00),
(34, 18, 1, 129.00), (34, 15, 1, 29.99), (34, 20, 1, 19.99),
(35, 17, 1, 79.99),
(36, 18, 1, 129.00), (36, 16, 1, 59.99), (36, 15, 2, 29.99), (36, 19, 3, 24.99),
(37, 2, 1, 199.00), (37, 3, 1, 49.99),
(38, 7, 2, 39.99), (38, 9, 5, 19.99), (38, 19, 2, 24.99),
(39, 11, 3, 45.00), (39, 12, 2, 55.00), (39, 14, 1, 50.00),
(40, 16, 1, 59.99), (40, 19, 3, 24.99), (40, 20, 2, 19.99);

CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_country ON orders(country);
CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON order_items(product_id);
