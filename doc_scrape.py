#1 - XPATH with Loop
#2 - Class Name

#xpath: er index starts from 1  -> i
#index number = string
#typecast - int(i) -> str(i) a convert kore kaj korte during loop

#//*[@id="root"]/div/div[2]/div[i]/div/div[1]/div[2]/div[1]/div/div/div[2]/div[2]/a - relative xpath

# absolute - /html/body/div[4]/div/div[2]/div[1]/div/div[1]/div[2]/div[1]/div/div/div[2]/div[2]/a


#step 1: Find out the pattern
from selenium import webdriver

from selenium.webdriver.common.by import By

import time

import csv

driver = webdriver.Chrome()

driver.get('https://heliumdoc.com/bd/doctors/bd/1/')

driver.maximize_window()

title_list = []

for page in range(30,36):
       driver.get(f'https://heliumdoc.com/bd/doctors/bd/{page}/')
       
       
       for i in range(1,14):
        j = str(i)
        title = driver.find_element(By.XPATH,'//*[@id="listings"]/div[3]/div/div[1]/div/div[2]/div['+j+']/div[1]/a/div/div[2]').text 
        title_list.append(title)

with open('doctors_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow(['Name', 'Specialization', 'Hospital'])
    
    for doctor in title_list:
        # Split the doctor info by newlines
        details = doctor.split('\n')
        # Extract the components (first 3 parts)
        name = details[0] if len(details) > 0 else ''
        specialization = details[1] if len(details) > 1 else ''
        hospital = details[3] if len(details) > 3 else ''  # Note: Hospital is at index 3
        writer.writerow([name, specialization, hospital])

print("Data successfully saved to doctors_data.csv with separate columns")
print(title_list)
print(len(title_list))
#page->all_products->page    