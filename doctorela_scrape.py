from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import csv

# Chrome options
chrome_options = Options()
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")

# Start browser
driver = webdriver.Chrome(options=chrome_options)
driver.get('https://doctorola.com/')
driver.maximize_window()
time.sleep(10)

# Locate the search input and search "Pulmonology"
search_input = driver.find_element(By.XPATH, '//*[@id="text"]')
search_input.send_keys("Oral ")
search_input.send_keys(Keys.RETURN)

# Wait for results to load
time.sleep(5)

# Scroll inside the doctor listing container
scrollable_div = driver.find_element(By.XPATH, '//*[@id="search_result"]')
last_height = driver.execute_script("return arguments[0].scrollHeight", scrollable_div)

for i in range(0, last_height + 1000, 100):
    driver.execute_script("arguments[0].scrollTop = arguments[1];", scrollable_div, i)
    time.sleep(0.4)

# Wait after scrolling
time.sleep(2)

# Get all doctor cards
doc_elements = driver.find_elements(By.XPATH, '//*[@id="search_result"]//a')

# Prepare data list
doctor_data = []

for elem in doc_elements:
    full_text = elem.text.strip()
    if full_text:
        lines = full_text.split('\n')
        name = lines[0] if len(lines) > 0 else ""
        speciality = lines[1] if len(lines) > 1 else ""
        hospital = lines[2] if len(lines) > 2 else ""
        doctor_data.append([name, speciality, hospital])

# Save to CSV
with open('cardio.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Speciality', 'Hospital'])
    writer.writerows(doctor_data)

print(f"✅ Saved {len(doctor_data)} doctors to cardio.csv")

# Optional: Keep browser open
time.sleep(30)
driver.quit()
