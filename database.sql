SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";

DROP DATABASE IF EXISTS billing;
CREATE DATABASE billing;
USE billing;

DROP USER IF EXISTS 'gsuser'@'localhost'; 
CREATE USER 'gsuser'@'localhost' IDENTIFIED BY 'gspass';
GRANT ALL PRIVILEGES ON billing.* TO 'gsuser'@'localhost';
FLUSH PRIVILEGES;

CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    weight VARCHAR(50),
    quantity_instock INT NOT NULL,
    price DECIMAL(10, 2) NOT NULL
);

INSERT INTO products (product_name, weight, quantity_instock, price) VALUES
('Cadbury 5Star 18g', '18g', 100, 10.00),
('Dairy Milk 12g', '12g', 100, 10.00),
('Himalayan water bottle 500ml', '500ml', 50, 25.00),
('Kissan Tomato ketchup 200g', '200g', 30, 85.00),
('Maggi Hot and sweet sauce 200g', '200g', 25, 75.00),
('Sofit soya flavoured drink 180ml', '180ml', 60, 15.00),
('Tresemme Conditioner 190ml', '190ml', 40, 169.00),
('Tresemme Shampoo 1L', '1L', 20, 845.00),
('Tresemme Shampoo 580ml', '580ml', 15, 499.00),
('Vaseline Aloe Fresh moisturiser 400ml', '400ml', 35, 450.00);


select * from products;