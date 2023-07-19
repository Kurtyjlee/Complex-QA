from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import time
from bs4 import BeautifulSoup
import random
import tqdm
import glob
from PyPDF2 import PdfReader
import pandas as pd

"""
Using url manipulation to download pdfs from RSIS. 
The text from the pdf are extracted in rsis_pdf_to_json.ipynb
"""

data_rsis_dir = "../data/RSIS"
get_url_fails = 0
download_pdf_fails = 0
extract_text_fails = 0

options = Options()
options.add_experimental_option('prefs', {
    "download.default_directory" : data_rsis_dir,
    "download.prompt_for_download" : False,
    "plugins.always_open_pdf_externally" : True
})
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.maximize_window()

title_list = []

# Getting all links from n pages
print("Getting page urls...")
n = 100
progress_bar = tqdm.tqdm(total=n)
for i in range(1, n):
    try:
        url = f"https://www.rsis.edu.sg/publications/rsis-publications/commentaries/page/{i}"

        # Accessing the webpage
        driver.get(url)
        page_source = driver.page_source

        soup = BeautifulSoup(page_source, 'lxml')

        # Getting all post urls
        titles = soup.find_all('p', class_='title title-box')
        for title in titles:
            title_text = title.find('a', class_='link')["href"]
            title_list.append(title_text)
        
        # rest randomly
        max_rest_time = 3
        random_number = random.randint(1, max_rest_time)
        time.sleep(random_number)
    except:
        get_url_fails += 1

    progress_bar.update(1)

# cycling through all the title_list
print("Getting pdfs...")
progress_bar = tqdm.tqdm(total=len(title_list))
for url in title_list:
    try:
        driver.get(url)
        driver.implicitly_wait(10)
        time.sleep(1)

        pdf_link = driver.find_element(By.XPATH, '/html/body/div[3]/section/div/div/div[2]/div[2]/div[1]/div/div[2]/div/div/a').click()

        # rest randomly
        time.sleep(5)
    except:
        download_pdf_fails += 1

    progress_bar.update(1)

# Getting text from all pdf
print("extracting text from pdf")
file_list = glob.glob(data_rsis_dir + '/*')
progress_bar = tqdm.tqdm(total=len(file_list))
text_list = []
for file in file_list:
    try:
        reader = PdfReader(file)
        text = ""
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text += page.extract_text()
        text_list.append(text)
    except:
        extract_text_fails += 1
    
    progress_bar.update(1)

print("converting to csv of pdfs")
# Converting to a csv
df = pd.DataFrame(text_list)

output = "../data/"
df.to_csv(f"{output}/output/rsis.csv")

print(f"total fails: {get_url_fails}, {download_pdf_fails}, {extract_text_fails}")
