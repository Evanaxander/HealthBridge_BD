from selenium import webdriver

from selenium.webdriver.common.by import By

import time

driver = webdriver.Chrome()

driver.get('https://doctorola.com/profile/24/Prof.-Dr.-Manzoor-Kader/1753967094/8/1/3')

driver.maximize_window()



title = driver.find_element(By.XPATH,'//*[@id="doctor_profile_div"]/div/div/div/div[1]').text

link = driver.find_element(By.XPATH,'//*[@id="doctor_profile_div"]/div/div/div/div[1]').get_attribute('href')

print(title,link)

time.sleep(20)