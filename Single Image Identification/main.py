# from keras.models import load_model  # TensorFlow is required for Keras to work
# import cv2  # Install opencv-python
# import numpy as np
# import re

# # Disable scientific notation for clarity
# np.set_printoptions(suppress=True)

# # Load the model
# model = load_model("keras_Model.h5", compile=False)

# # Load the labels
# with open("labels.txt", "r") as file:
#     class_names = [line.strip() for line in file.readlines()]

# # Dictionary to hold product prices
# product_prices = {
#     "Cadbury 5Star 18g": 10.00,
#     "Dairy Milk 12g": 10.00,
#     "Himalayan water bottle 500ml": 25.00,
#     "Kissan Tomato ketchup 200g": 85.00,
#     "Maggi Hot and sweet sauce 200g": 75.00,
#     "Sofit soya flavoured drink 180ml": 15.00,
#     "Tresemme Conditioner 190ml": 169.00,
#     "Tresemme Shampoo 1L": 845.00,
#     "Tresemme Shampoo 580ml": 499.00,
#     "Vaseline Aloe Fresh moisturiser 400ml": 450.00
# }

# # Function to process and predict an uploaded image
# def process_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)

#     # Check if the image was loaded successfully
#     if image is None:
#         print("Error: Unable to load the image. Check the file path.")
#         return

#     # Resize the image to the model's input size (224x224)
#     image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

#     # Convert the image to a numpy array and reshape it for the model's input
#     image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

#     # Normalize the image array
#     image = (image / 127.5) - 1

#     # Predict using the model
#     predictions = model.predict(image)
    
#     detected_classes = []
#     confidence_scores = []

#     # Iterate through each class prediction
#     for index in range(len(class_names)):
#         confidence_score = predictions[0][index]
#         if confidence_score > 0.2:  # Check if confidence score is above the threshold
#             class_name = re.sub(r'^\d+\s*', '', class_names[index].strip())  # Get the product name
#             detected_classes.append(class_name)
#             confidence_scores.append(confidence_score)

#             # Print detected class and its confidence score
#             print("Detected:", class_name)
#             print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

#     # Display prices for detected classes
#     total_bill = 0  # Initialize total bill
#     for detected_class in detected_classes:
#         price = product_prices.get(detected_class, "Price not available")
#         if isinstance(price, (int, float)):  # Ensure price is a number
#             total_bill += price  # Add price to total bill
#         print(f"Price of {detected_class}: Rs{price}")

#     # Display the total bill
#     print("-------------------------------------------------")
#     print(f"\nTotal Bill: Rs{total_bill:.2f}")
#     print("-------------------------------------------------")


# # Provide the path to the image you want to upload
# image_path = r"C:\Varnitha1\Archive\Vaseline Aloe Fresh moisturiser 400ml\IMG_4116.jpg" # Replace with your image file path
# process_image(image_path)
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from keras.models import load_model
import cv2
import numpy as np
import re
import qrcode
from PIL import Image, ImageTk

# Load the model and class names
model = load_model("keras_Model.h5", compile=False)
with open("labels.txt", "r") as file:
    class_names = [line.strip() for line in file.readlines()]

# Dictionary for product prices
product_prices = {
    "Cadbury 5Star 18g": 10.00,
    "Dairy Milk 12g": 10.00,
    "Himalayan water bottle 500ml": 25.00,
    "Kissan Tomato ketchup 200g": 85.00,
    "Maggi Hot and sweet sauce 200g": 75.00,
    "Sofit soya flavoured drink 180ml": 15.00,
    "Tresemme Conditioner 190ml": 169.00,
    "Tresemme Shampoo 1L": 845.00,
    "Tresemme Shampoo 580ml": 499.00,
    "Vaseline Aloe Fresh moisturiser 400ml": 450.00
}

# List to store all detected products
all_detected_products = []

# Function to process and predict the uploaded image
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        result_label.config(text="Error: Unable to load the image.")
        return

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    predictions = model.predict(image)
    detected_products = []
    
    for index, score in enumerate(predictions[0]):
        if score >= 0.2:
            class_name = re.sub(r'^\d+\s*', '', class_names[index].strip())
            # Check if product is in the database
            if class_name in product_prices:
                detected_products.append((class_name, score))
            else:
                messagebox.showinfo("Product Not Found", f"'{class_name}' is not in the database. Please check the product.")

    # Update the global list and display products
    all_detected_products.extend(detected_products)
    display_products(all_detected_products)

# Display products in the table and calculate the total bill
def display_products(detected_products):
    for row in table.get_children():
        table.delete(row)

    total_bill = 0
    product_count = {}

    for product_name, confidence in detected_products:
        price = product_prices.get(product_name, "N/A")
        if price != "N/A":  # Only add products with valid prices
            if product_name not in product_count:
                product_count[product_name] = {'quantity': 0, 'price': price}
            product_count[product_name]['quantity'] += 1

    for product_name, details in product_count.items():
        quantity = details['quantity']
        price = details['price']
        total_price = price * quantity
        total_bill += total_price
        table.insert("", "end", values=(product_name, f"Rs {price}", quantity, f"Rs {total_price}"))

    total_amount.set(f"Total Amount: Rs {total_bill:.2f}")


from datetime import datetime

# Function to generate and display the receipt
def display_receipt(final_amount):
    receipt_window = tk.Toplevel(window)
    receipt_window.title("Receipt")
    receipt_window.geometry("800x600")  # Adjust dimensions for better readability

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add a scrollable text area
    text_area = tk.Text(receipt_window, wrap="word", font=("Courier", 10))  # Courier font is monospaced for alignment
    text_area.pack(fill="both", expand=True)

    # Header
    receipt_text = "\t\t\tGrocery Billing System\n\n\n"
    receipt_text += f"Date & Time: {current_datetime}\n\n"
    receipt_text += f"{'Product Name':<40} | {'Quantity':<10} | {'Price':<10} | {'Discount':<10} | {'Final Price':<10}\n"
    receipt_text += "-"*85 + "\n"

    # Table content
    for item in table.get_children():
        product_name, price, quantity, total_price = table.item(item, 'values')
        receipt_text += f"{product_name:<40} | {quantity:<10} | {price:<10} | {'Rs 0.00':<10} | {total_price:<10}\n"
    
    # Footer with final total
    receipt_text += "-"*85 + "\n"
    receipt_text += f"Final Total Bill: {final_amount}\n"
    receipt_text += "-"*85 + "\n\n\n\n"
    receipt_text += "\t\tThank You for shopping, please visit again!\n"

    # Insert receipt text into text area and disable editing
    text_area.insert("1.0", receipt_text)
    text_area.config(state="disabled")

    # Optionally, show a message box with the final bill
    messagebox.showinfo("Receipt", receipt_text)

# Function to delete a selected product from the table
def delete_selected_product():
    selected_item = table.selection()
    if selected_item:
        for item in selected_item:
            product_name = table.item(item, 'values')[0]
            all_detected_products[:] = [(pname, score) for pname, score in all_detected_products if pname != product_name]
            table.delete(item)
        display_products(all_detected_products)
    else:
        messagebox.showwarning("Delete Product", "Please select a product to delete.")

# Open file dialog to upload image
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        process_image(file_path)

# Generate UPI QR code with the exact amount
def generate_upi_qr_code(amount):
    upi_id = "varnitharaghunath14@ybl"
    qr_data = f"upi://pay?pa={upi_id}&pn=Store&am={amount:.2f}"
    qr = qrcode.make(qr_data)
    qr.show()

# Show payment options only after generating the final bill
def show_payment_options():
    amount = float(total_amount.get().split("Rs")[-1].strip())
    if amount > 0:
        coupon_code = simpledialog.askstring("Apply Coupon", "Do you want to apply a coupon?")
        if coupon_code == "BATCH69":
            discount = amount * 0.22
            amount -= discount
            messagebox.showinfo("Coupon Applied", "Successfully applied coupon. You have got 22% off on the final bill.")
        payment_frame.pack(pady=10)
        total_amount.set(f"Total Amount: Rs {amount:.2f}")
        generate_qr_button.config(command=lambda: generate_upi_qr_code(amount))
    else:
        messagebox.showwarning("Generate Bill", "No items in the bill. Add products first.")

# Display thank you message on cash payment
def cash_payment():
    final_amount = total_amount.get()
    display_receipt(final_amount)
    clear_all()

# Function to clear all products and reset for new billing
def clear_all():
    all_detected_products.clear()
    for row in table.get_children():
        table.delete(row)
    total_amount.set("Total Amount: Rs 0.00")
    payment_frame.pack_forget()

# GUI Window setup
window = tk.Tk()
window.title("Grocery Billing System")
window.geometry("600x700")
window.configure(bg="#f5f5f5")

# Title label
title_label = tk.Label(window, text="Grocery Billing System", font=("Times New Roman", 22, "bold"), fg="blue", padx=20, pady=10)
title_label.pack(pady=10)

# Frame for product table
table_frame = tk.Frame(window, bg="#f5f5f5")
table_frame.pack(pady=10)

# Product table setup
table = ttk.Treeview(table_frame, columns=("Product Name", "Price (per unit)", "Quantity", "Total"), show="headings", height=8)
table.heading("Product Name", text="Product Name")
table.heading("Price (per unit)", text="Price (per unit)")
table.heading("Quantity", text="Quantity")
table.heading("Total", text="Total")
table.column("Product Name", width=200)
table.column("Price (per unit)", width=100)
table.column("Quantity", width=80)
table.column("Total", width=100)
table.pack()

# Upload and Delete buttons
upload_button = tk.Button(window, text="Choose File", command=upload_image, font=("Arial", 10), bg="#4CAF50", fg="white")
upload_button.pack(pady=5)
delete_button = tk.Button(window, text="Delete Selected Product", command=delete_selected_product, font=("Arial", 10), bg="#e53935", fg="white")
delete_button.pack(pady=5)

# Billing summary section
billing_frame = tk.Frame(window, bg="#f5f5f5")
billing_frame.pack(pady=10)

total_amount = tk.StringVar(value="Total Amount: Rs 0.00")
total_label = tk.Label(billing_frame, textvariable=total_amount, font=("Times New Roman", 15, "bold"), bg="#f5f5f5")
total_label.pack()

# Button to generate final bill
generate_bill_button = tk.Button(billing_frame, text="Generate Final Bill", command=show_payment_options, font=("Arial", 13), bg="green", fg="white")
generate_bill_button.pack(pady=10)

# Payment options frame (initially hidden)
payment_frame = tk.Frame(window, bg="#f5f5f5")
cash_button = tk.Button(payment_frame, text="Cash", font=("Arial", 12), width=10, bg="#2196F3", fg="black", command=cash_payment)
cash_button.grid(row=0, column=0, padx=5)
card_button = tk.Button(payment_frame, text="Card", font=("Arial", 12), width=10, bg="#2196F3", fg="black", command=lambda: messagebox.showinfo("Card Payment", "Please proceed with card payment at the counter."))
card_button.grid(row=0, column=1, padx=5)
generate_qr_button = tk.Button(payment_frame, text="UPI", font=("Arial", 12), width=10, bg="#2196F3", fg="black")
generate_qr_button.grid(row=0, column=2, padx=5)

# Clear button to reset all data
clear_button = tk.Button(window, text="Clear", command=clear_all, font=("Arial", 12), bg="orange", fg="white")
clear_button.pack(pady=10)

# Start the Tkinter event loop
window.mainloop()